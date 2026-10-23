"""Write completion report from verified evidence, never held-out outcomes."""
import json
from pathlib import Path
import audit_phase4 as audit

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'outputs/phase4_v3_release_quality';data=ROOT/'data/phase4_v3_release'
    exp=ROOT/'outputs/phase4_v3_expansion'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    gate=read(exp/'gate.json');selection=read(exp/'selection.json')
    verified=read(out/'verification.json');state=read(data/'generation_state.json')
    if state['status']!='verified_complete' or verified['failed_partitions']:
        raise ValueError('Cannot report completion before verification')
    checks=read(exp/'joint_comparison_results.json');manifest=read(data/'dataset_manifest.json')
    exclusions=read(out/'train_exclusions.json')
    totalbytes=sum(f['bytes'] for f in manifest['files'])
    lines=['# Phase 4 검증 확대와 v3 전체 데이터 완료 보고','',
        '범위: 수치 검증 확대와 새 버전 데이터 생성·기계적 검증까지. 머신러닝 학습, 포스터용 분석 갱신, GitHub 업로드는 하지 않았다.','',
        '## 1. 새로운 train 수치 검증','',
        f"기존 19개와 겹치지 않는 {gate['scenario_count']}개 시나리오를 seed={selection['seed']}로 고정 선정했다. 총 {gate['candidates']:,}개 후보다.",
        '유형×장애물 배치별 사례, 작은 여유·특이점 근처·도달 경계·도달 불가 후보를 포함했다. 전체 시나리오에서 42개 후보를 모두 검사했다. 결과가 좋은 사례만 뽑은 검증은 아니다.',
        f"경로·적분 상태: {gate['candidate_status_counts']}. L/Kε/O의 161·321·641점 비교 실패: {gate['geometry_failure_counts'] or '없음'}.",
        f"Eθ/A/J의 적응형 값 대 독립 미분 2561점 적분 비교: {gate['joint_comparisons']:,}건 중 {gate['joint_comparison_failures']}건 불일치.",
        '', '|관절 지표|최대 절대 차이|최대 상대 차이|','|---|---:|---:|']
    for field in gate['max_joint_relative_error']:
        group=[c for c in checks if c['field']==field]
        lines.append(f"|{field}|{max(c['absolute_error'] for c in group):.8g}|{max(c['relative_error'] or 0 for c in group):.8g}|")
    lines+=['','2561점도 참값이 아니다. 해석 미분의 검증 사례, 적응형 적분기의 오차 추정, 독립 가중 재적분, 격자 비교를 함께 확인했다. 엄밀한 정확도 보증으로 일반화하지 않는다.',
        '', '## 2. 생성된 새 버전 데이터','',
        '버전: phase4-mathematics-v3 / release-generator-v1. 원 시나리오 입력과 scenario_id/split을 보존했다.',
        '', '|분할|시나리오|후보|시간표본|목적별 라벨|','|---|---:|---:|---:|---:|']
    for split,r in verified['split_counts'].items():
        lines.append(f"|{split}|{r['scenarios']:,}|{r['candidates']:,}|{r['samples']:,}|{r['winners']:,}|")
    lines+=['',f"완료 데이터 파일 {len(manifest['files']):,}개, 압축 합계 약 {totalbytes/1024**3:.2f} GiB. 원자료·이전 파일럿과 다른 폴더에 저장했다.",
        f"생성 경과시간 약 {(state['last_update_utc']-state['created_time_utc'])/60:.2f}분, 최종 검증 {verified['elapsed_seconds']:.2f}초. 생성 경과시간에는 중간 대화 중단 중 실행된 시간도 포함된다.",
        '관절 지표는 독립 미분+적응형 구간 GL32 가중합이다. 161개 시간표본과 별도 적분 노드/가중치/피적분함수를 저장했다.',
        'L/Kε/O는 원 161점 값을 유지하면서 각 계산 가능한 후보에 321·641점 수렴성 검사를 추가했다. 실패 후보는 unresolved로 분리한다. 식과 단위·epsilon·종합 가중치·동점 규칙은 변경하지 않았다.',
        'accepted / invalid / unresolved를 구분한다. 유효 후보가 없는 상황은 42개 후보군에서의 실패이지 환경 자체의 경로 부재가 아니다. 미확정 대안이 있으면 승자 라벨도 잠정으로 표시한다.',
        '', '## 3. 실제 수행한 검증','',
        f"시나리오별 후보·시간표본·라벨 수, ID 유일성, 결합 키, 분할, 곡선식 스키마, 파일 크기·해시: {verified['partition_count']:,}개 파티션에서 실패 없음.",
        f"train 관절 가중 재적분 최대 절대 차이: {verified['train_replay_max']:.8g}. train 순기구학 복원 최대 오차: {verified['train_fk_max']:.8g}.",
        f"train 후보 상태: {verified['train_status_counts']}. 이 수치는 수치 검증의 범위/제외 상태이며 곡선 우열 분석이 아니다.",
        f"미확정 원인: {exclusions['reasons']}. 실패한 L/Kε/O 검사: {exclusions['failed_geometry_counts']}. 세부 run_id·검산값·목격 위치는 train_exclusions.json에 있다.",
        f"구 계산의 표본 검사에서 유효였으나 새 구간 검사에서 위반을 발견한 train 후보: {exclusions['newly_invalid_count']}개. 해당 구버전 유효 라벨을 새 버전에 그대로 가져오지 않았다.",
        'train에서 정규화·점수·동점·승자 재계산도 확인했다. validation/test/challenge는 ID·행 수·스키마·분할·해시만 확인했다. held-out 평가값 분포, 유효율, 곡선 순위, 승자 비율을 집계하지 않았다.',
        f"기존 분석 산출물 보존: {verified['previous_analysis_preserved']}. 이전 v3 파일럿 보존: {verified['pilot_preserved']}.",
        '회귀 테스트: '+' / '.join(verified['regression_tests'][-3:]),
        '', '## 4. 한계와 다음 단계','',
        '데이터 생성/무결성 완료는 추천 모델의 성능 보증이나 모든 입력에서의 수학적 정확도 보증이 아니다. 후보별 검증 상태와 잠정 라벨을 분석/학습 단계에서 반드시 반영해야 한다.',
        '이번에는 데이터 수를 늘리면 과적합이 해결된다고 주장하지 않는다. 같은 scenario_id의 후보는 독립 표본이 아니며 train/validation/test 경계를 유지한다.',
        '다음 단계는 새 train 자료의 표·그래프 갱신이다. 그 이후 후보 추천 모델을 학습하고 validation으로 선택하며 test/challenge는 마지막 평가에 사용한다.',
        '', '## 파일 및 재실행','',
        f'데이터: {data}',f'검증 확대: {exp}',f'완료 검증: {out}',
        '데이터 사전: docs/phase4_release_data_dictionary.md. 사전 정책: docs/phase4_release_policy.md.',
        '검증 시 Python/라이브러리 버전은 verification_environment.json. 재현/재개 시 같은 환경을 유지한다.',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe -m pip install -e ".[data,analysis,refinement]"',
        '.\\.venv\\Scripts\\python.exe scripts/phase4_release_pipeline.py expand --workers 6 --resume',
        '.\\.venv\\Scripts\\python.exe scripts/verify_phase4_release.py expand --workers 6',
        '.\\.venv\\Scripts\\python.exe scripts/phase4_release_pipeline.py full --workers 6 --resume',
        '.\\.venv\\Scripts\\python.exe scripts/verify_phase4_release.py full --workers 6',
        '.\\.venv\\Scripts\\python.exe scripts/summarize_release_exclusions.py',
        '.\\.venv\\Scripts\\python.exe scripts/report_phase4_release.py',
        '```','',
        '완료된 데이터는 다시 생성할 필요 없다. 재개는 동결된 소스·정책·입력 해시가 같을 때만 허용하며 완료 파티션을 검증 후 건너뛴다. 미완료 attempt는 지우지 않고 다른 attempt에 재시도한다.']
    (out/'release_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit.write_json(out/'output_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=audit.sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='output_manifest.json']))
    print('Report:',out/'release_report.md')


if __name__=='__main__':main()
