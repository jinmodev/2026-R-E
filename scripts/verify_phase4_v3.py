"""Stream v3 pilot verification; reports outside all source datasets."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq
import audit_phase4 as audit
from robot_planner.kinematics import PlanarArm2R
from robot_planner.phase4 import score_phase4_candidates,select_scenario_winners
from robot_planner.phase4_v3 import DATASET_VERSION,JOINT_FIELDS
from robot_planner.refinement import differential_joint_state
from refine_phase4 import objects

ROOT=Path(__file__).resolve().parents[1]


def lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def verify(data,out):
    if data.resolve()==out.resolve() or data.resolve() in out.resolve().parents:
        raise ValueError('Verification reports must be outside the dataset')
    out.mkdir(parents=True,exist_ok=True);checks=[]
    def check(name,ok,detail=None):checks.append(dict(check=name,passed=bool(ok),detail=detail))
    meta=json.loads((data/'metadata.json').read_text(encoding='utf-8'))
    check('complete v3 train dataset',meta['status']=='complete' and meta['dataset_version']==DATASET_VERSION and meta['data_split']=='train')
    for rel,digest in meta['hashes'].items():
        check('input/source hash '+rel,audit.sha(ROOT/rel)==digest)
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=data/item['path']
        check('data hash/size '+item['path'],path.stat().st_size==item['bytes'] and audit.sha(path)==item['sha256'])
    records=list(lines(data/'trajectory_metrics.jsonl'));index={r['run_id']:r for r in records}
    check('run count/unique IDs',len(index)==len(records)==meta['run_count'])
    check('train-only and version',all(r['data_split']=='train' and r['dataset_version']==DATASET_VERSION for r in records))
    check('scenario candidate count',set(Counter(r['scenario_id'] for r in records).values())=={42})
    check('source selection IDs',set(r['scenario_id'] for r in records)==set(meta['scenario_ids']))
    check('equations present',all(all(r.get(k) for k in ('x_t_equation','y_t_equation','theta1_t_equation','theta2_t_equation')) for r in records))
    sums=defaultdict(list);ranges=defaultdict(list);panels=0;nodes=0;node_error=0.;panel_error=0.;time_ok=True;key_ok=True
    active=None;obj=None;lookup={v[0]:k for k,v in JOINT_FIELDS.items()}
    for p in lines(data/'adaptive_panels.jsonl'):
        rid=p['run_id'];panels+=1;nodes+=len(p['time_s'])
        key_ok=key_ok and rid in index and p['scenario_id']==index[rid]['scenario_id'] and p['data_split']=='train'
        if rid!=active:
            active=rid;obj=objects(index[rid])
        arm,curve,_=obj;row=index[rid]
        time_ok=time_ok and len(p['time_s'])==len(p['weights_s'])==len(p['integrand'])==32
        time_ok=time_ok and 0<=p['lo']<p['hi']<=row['duration_s']
        time_ok=time_ok and all(p['lo']<t<p['hi'] for t in p['time_s']) and all(w>0 for w in p['weights_s'])
        time_ok=time_ok and math.isclose(math.fsum(p['weights_s']),p['hi']-p['lo'],rel_tol=1e-12,abs_tol=1e-12)
        recomputed=math.fsum(w*v for w,v in zip(p['weights_s'],p['integrand']))
        panel_error=max(panel_error,abs(recomputed-p['integral']))
        # Check first/middle/last nodes on EVERY saved panel, not only convenient cases.
        for j in (0,16,31):
            state=differential_joint_state(arm,curve,p['time_s'][j],row['duration_s'],row['elbow_mode'])
            truth=math.fsum(v*v for v in state[lookup[p['metric']]])
            node_error=max(node_error,abs(truth-p['integrand'][j])/max(1,abs(truth)))
        if p['integration_status']=='accepted':
            sums[(rid,p['metric'])].append(recomputed);ranges[(rid,p['metric'])].append((p['lo'],p['hi']))
    check('adaptive panel/node count',panels==meta['adaptive_panel_count'] and nodes==meta['adaptive_node_count'])
    check('panel join keys',key_ok);check('panel domain/weights',time_ok)
    check('weighted panel replay',panel_error==0,dict(max_absolute_error=panel_error))
    check('integrand recalculation 3 nodes/every panel',node_error<1e-12,dict(max_scaled_error=node_error,nodes_checked=3*panels))
    summary_error=0.;coverage_ok=True;eligibility_ok=True
    for row in records:
        accepted=row['candidate_status']=='accepted'
        eligibility_ok=eligibility_ok and row['selection_valid']==accepted
        if accepted:
            eligibility_ok=eligibility_ok and row['path_check_status']=='bounded_clear' and row['joint_integration_status']=='accepted'
        for field in lookup:
            if accepted:
                key=(row['run_id'],field);parts=sorted(ranges[key])
                coverage_ok=coverage_ok and bool(parts) and parts[0][0]==0 and parts[-1][1]==row['duration_s']
                coverage_ok=coverage_ok and all(a[1]==b[0] for a,b in zip(parts,parts[1:]))
                summary_error=max(summary_error,abs(math.fsum(sums[key])-row[field]))
            else:eligibility_ok=eligibility_ok and row[field] is None
    check('candidate eligibility / null scores',eligibility_ok)
    check('accepted panel coverage no gaps/overlap',coverage_ok)
    check('summary weighted replay',summary_error==0,dict(max_absolute_error=summary_error))
    samplecounts=Counter();samplekeys=set();fkmax=0.;sample_ok=True
    for path in sorted((data/'trajectory_samples').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048,columns=[
            'run_id','scenario_id','data_split','sample_index','time_s','x','y','theta1_rad','theta2_rad']):
            for r in batch.to_pylist():
                rid=r['run_id'];samplecounts[rid]+=1;key=(rid,r['sample_index'])
                sample_ok=sample_ok and key not in samplekeys and rid in index and r['data_split']=='train'
                samplekeys.add(key);base=index[rid]
                sample_ok=sample_ok and r['scenario_id']==base['scenario_id'] and 0<=r['sample_index']<161
                sample_ok=sample_ok and math.isclose(r['time_s'],r['sample_index']*base['duration_s']/160,abs_tol=1e-12)
                if r['theta1_rad'] is not None and r['theta2_rad'] is not None:
                    arm=PlanarArm2R(base['link1'],base['link2'],(base['base_x'],base['base_y']))
                    end=arm.forward(r['theta1_rad'],r['theta2_rad']).end
                    fkmax=max(fkmax,math.dist(end,(r['x'],r['y'])))
    check('sample count/join/index/time',sample_ok and set(samplecounts)==set(index) and set(samplecounts.values())=={161}
          and sum(samplecounts.values())==meta['trajectory_sample_count'])
    check('theta forward kinematics',fkmax<1e-9,dict(max_absolute_error=fkmax))
    recomputed=score_phase4_candidates(records)
    check('score recomputation',all(math.isclose(a['balanced_cost_0_100'],b['balanced_cost_0_100'],rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(records,recomputed)))
    winners=list(lines(data/'scenario_winners.jsonl'));actual={(r['scenario_id'],r['objective']):r for r in winners}
    expected=select_scenario_winners(recomputed)
    check('winner IDs/ties recomputation',len(actual)==len(winners)==len(expected)==meta['winner_count'] and all(
        actual[(r['scenario_id'],r['objective'])]['winner_run_id']==r['winner_run_id'] and
        actual[(r['scenario_id'],r['objective'])]['tied_run_ids_json']==r['tied_run_ids_json'] for r in expected))
    check('all labels explicitly pilot-only',all(r['label_readiness'] in ('pilot_only_not_training_approved','provisional_unresolved_candidates') for r in winners))
    old=json.loads((ROOT/'outputs/phase4_quality/output_manifest.json').read_text(encoding='utf-8'))
    check('previous analysis artifacts preserved',all((ROOT/r['path']).stat().st_size==r['bytes'] and audit.sha(ROOT/r['path'])==r['sha256'] for r in old['outputs']),len(old['outputs']))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (out/'unit_test_log.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    check('all regression tests',tests.returncode==0,tests.stderr.splitlines()[-4:])
    summary=dict(passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks),checks=checks)
    audit.write_json(out/'verification.json',summary)
    changes=json.loads((data/'label_changes.json').read_text(encoding='utf-8'))
    report=['# Phase 4 v3 파일럿 결과','',
        f"버전: {meta['dataset_version']}. train {meta['scenario_count']}개 시나리오, 후보 {meta['run_count']}개, 시간표본 {meta['trajectory_sample_count']:,}개.",
        f"상태: {meta['status_counts']}. 총 생성 {meta['elapsed_seconds']:.2f}초.",
        f"적응형 적분 구간 {panels:,}개, 저장 적분점 {nodes:,}개. 161 시간표본과 별도 저장.",
        f"검증 {summary['passed']}개 통과, {summary['failed']}개 실패. 가중 재적분 최대 절대오차 {summary_error:.6g}, FK 최대 오차 {fkmax:.6g}.",
        f"v2 대비 결정론적 승자 변경 {sum(r['winner_changed'] for r in changes)}건, 동점집합 변경 {sum(r['tie_set_changed'] for r in changes)}건 / {len(changes)}개 시나리오-목적 조합.",
        '', '## 바뀐 계산','',
        'Eθ/A/J는 독립 관절 미분과 적응형 구간 적분이다. 저장한 weights_s와 integrand의 가중합이 최종 점수다.',
        'L/Kε/O는 기존 표본 적분을 명시적으로 유지했다. 161개 시간표본을 사다리꼴 적분한 관절 지표는 진단용이며 최종 점수와 다를 수 있다.',
        'QUADPACK과 최종 32점 Gauss-Legendre 재적분의 일치 및 오차 추정치를 확인했다. 이는 엄밀한 정확도 인증은 아니다.',
        'invalid는 관절 적분을 생략하고 null을 저장한다. unresolved는 유효 후보로 승인하지 않으며 대안 중 unresolved가 있으면 승자도 잠정 표시한다.',
        '', '## 보존·해석·다음 단계','',
        '기존 GUI와 v2 생성기/데이터를 변경하지 않았다. 곡선식·theta(t) 합성식은 v3 데이터에도 포함된다.',
        '기존 19개 대표 train 사례의 파일럿이다. 무작위 전체 모집단 분석도, 머신러닝 학습도 아니다. validation/test/challenge 평가값은 읽지 않았다.',
        '후보 42개 안에서의 비교일 뿐 전역 최적이나 과적합 방지 보증이 아니다. 기존 데이터와는 (dataset_version, run_id)로 구별한다.',
        '전체 생성 전에 더 넓은 고정 train 파일럿으로 L/Kε/O와 추가 특이점·미확정 처리를 검증해야 한다. 이번 속도를 전체 데이터에 그대로 외삽하지 않는다.',
        '반도체/모터의 실제 에너지 모델로 바뀐 것이 아니다. Eθ는 여전히 관절 운동 노력도다.',
        '', '## 파일과 실행','',
        f'데이터: {data.resolve()}',f'검증: {out.resolve()}',
        '정책: docs/phase4_v3_policy.md. 의존성·입력 해시·설정은 데이터 metadata.json.',
        '```powershell', '.\\.venv\\Scripts\\python.exe -m pip install -e ".[data,analysis,refinement]"',
        '.\\.venv\\Scripts\\python.exe scripts/generate_phase4_v3.py --output data/phase4_v3_pilot_new',
        '.\\.venv\\Scripts\\python.exe scripts/verify_phase4_v3.py --data data/phase4_v3_pilot_new --output outputs/phase4_v3_quality_new',
        '```','', '생성기는 비어 있지 않은 출력 폴더를 거부한다. 기존 파일럿은 재생성 없이 검증 명령만 실행할 수 있다.']
    (out/'pilot_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'},ensure_ascii=False))
    for c in checks:
        if not c['passed']:print(c)
    if summary['failed']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/phase4_v3_pilot')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/phase4_v3_quality')
    args=parser.parse_args();verify(args.data,args.output)
