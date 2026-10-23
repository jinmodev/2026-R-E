"""Reproduce the user's exported case, preserving inputs; no held-out data."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, shutil, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from robot_planner.report_evaluation import compare_candidates, score_report_candidates
from robot_planner.report_physics import ReportPhysicsConfig
from robot_planner.kinematics import PlanarArm2R
from robot_planner.evaluation import CircleObstacle
from robot_planner.curves import make_curve
from robot_planner.report_math import polynomial_coefficients

NAMES={'distance':'거리','joint_effort':'관절 노력도','acceleration':'각가속도',
       'jerk':'관절 저크','curvature':'정칙화 곡률','obstacle_penalty':'접근 벌점',
       'model_energy':'모형 에너지','rms_torque':'RMS 토크','report_balanced':'종합점수'}
FAMILIES={'line':'직선','arc':'원호','bezier':'베지어','hermite':'에르미트','cycloid':'사이클로이드','limacon':'리마송'}
COLORS={'line':'#64748b','arc':'#3563e9','bezier':'#e09225','hermite':'#8155b9','cycloid':'#d94d88','limacon':'#079c92'}
FIELDS=['run_id','curve_key','shape','elbow_mode','candidate_status','selection_valid',
        'energy_selection_ready','torque_selection_ready','report_scenario_labels_ready',
        'path_length_units','joint_velocity_effort','joint_acceleration_effort','joint_jerk_effort',
        'regularized_curvature_energy_inv_unit','obstacle_penalty_s_per_unit2',
        'min_obstacle_clearance_units','model_electrical_energy_j','rms_joint_torque_nm',
        'report_balanced_reward_0_100']

def read(path):
    def typed(v):
        if v=='': return None
        if v in ('true','false'): return v=='true'
        try: return float(v)
        except ValueError: return v
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [{k:typed(v) for k,v in r.items()} for r in csv.DictReader(f)]

def write(path,rows):
    rows=list(rows)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows: w.writerow({k:('true' if v else 'false') if isinstance(v,bool) else v for k,v in r.items()})

def sha(p): return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def close(a,b): return math.isclose(float(a),float(b),rel_tol=2e-10,abs_tol=1e-9)
def short(r): return r['run_id'].replace('interactive_','')
def label(r): return FAMILIES[r['curve_key']]+' '+('위' if r['elbow_mode']=='up' else '아래')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT/'분석자료_1')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/analysis_case1_20261021')
    ap.add_argument('--refresh',action='store_true',help='Refresh only this generated case output, never source inputs.');args=ap.parse_args()
    src=args.source;out=args.output
    assert not out.exists() or (args.refresh and (out/'verification.json').is_file()), 'Use a new output directory; no overwrite.'
    folders=list(src.glob('robot_path_comparison_*'));assert len(folders)==1
    folder=folders[0];j=json.loads((src/'robot_path_run.json').read_text(encoding='utf-8'))
    records=read(folder/'candidate_metrics.csv');labels=read(folder/'objective_winners.csv')
    equations=read(folder/'equations.csv');byid={r['run_id']:r for r in records};eq={r['run_id']:r for r in equations}
    source_files=[p for p in src.rglob('*') if p.is_file()]
    hashes={str(p.relative_to(src)):sha(p) for p in source_files}
    assert len(records)==42 and len(byid)==42 and len(eq)==42 and len(labels)==9
    assert all(r['scenario_id']=='interactive' for r in records)
    inp=j['inputs'];cfg=ReportPhysicsConfig(**j['physics_model'])
    for r in records:
        for k in ('base_x','base_y','link1','link2','start_x','start_y','goal_x','goal_y','duration_s'):
            assert close(r[k],inp[k]),(r['run_id'],k)
        assert json.loads(r['physics_config_json'])==j['physics_model']
    replay_labels=score_report_candidates(copy.deepcopy(records))
    for original,new in zip(labels,replay_labels):
        for k in ('objective','label_status','winner_run_id','tie_count','tied_run_ids_json'):
            assert original[k]==new[k],k
        if original['raw_metric_value'] is not None: assert close(original['raw_metric_value'],new['raw_metric_value'])
    arm=PlanarArm2R(inp['link1'],inp['link2'],(inp['base_x'],inp['base_y']))
    obstacles=[CircleObstacle(**o) for o in j['obstacles']]
    # Full fresh 42-candidate replay verifies the export against its actual inputs.
    replay=compare_candidates(arm,(inp['start_x'],inp['start_y']),(inp['goal_x'],inp['goal_y']),
                              obstacles=obstacles,duration_s=inp['duration_s'],physics_config=cfg)
    replay_by={r['run_id']:r for r in replay['records']}
    metric_fields=[k for k in FIELDS if k.endswith(('_units','_effort','_inv_unit','_s_per_unit2','_j','_nm','_0_100'))]
    for r in records:
        new=replay_by[r['run_id']]
        assert r['candidate_status']==new['candidate_status'] and r['selection_valid']==new['selection_valid']
        for k in metric_fields:
            assert (r[k] is None and new[k] is None) or close(r[k],new[k]),(r['run_id'],k)
    for old,new in zip(labels,replay['labels']):
        assert old['label_status']==new['label_status'] and old['winner_run_id']==new['winner_run_id']
    selected_ids=list(dict.fromkeys(r['winner_run_id'] for r in labels if r['label_status']=='ready'))
    selected=[byid[rid] for rid in selected_ids];valid=[r for r in records if r['selection_valid']]
    geometry=pd.read_csv(folder/'geometry_samples.csv',float_precision='round_trip')
    times=pd.read_csv(folder/'time_samples.csv',float_precision='round_trip')
    for frame in (geometry,times):
        assert not frame.duplicated(['run_id','sample_index']).any()
        assert set(frame.run_id)==set(byid)
        assert frame.groupby('run_id').size().eq(161).all()
    chosen=next(r for r in records if r['curve_key']==inp['curve_key'] and r['elbow_mode']==inp['elbow_mode'] and close(r['shape'],inp['shape']))
    assert close(chosen['path_length_units'],j['metrics']['path_length_units'])
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True);(out/'provenance_inputs').mkdir(exist_ok=True)
    for p in source_files:
        dest=out/'provenance_inputs'/p.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    clean=[{k:r[k] for k in FIELDS} for r in records]
    write(out/'all_42_candidates.csv',clean)
    write(out/'selected_5_candidates.csv',[{k:r[k] for k in FIELDS} for r in selected])
    write(out/'objective_winners.csv',labels)
    geometry[geometry.run_id.isin(selected_ids)].to_csv(out/'selected_geometry_samples.csv',index=False,encoding='utf-8-sig')
    times[times.run_id.isin(selected_ids)].to_csv(out/'selected_time_samples.csv',index=False,encoding='utf-8-sig')
    write(out/'selected_equations.csv',[eq[rid] for rid in selected_ids])
    font_path=Path('C:/Windows/Fonts/malgun.ttf');assert font_path.exists()
    font_manager.fontManager.addfont(str(font_path));plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(out/'figures'/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
        fig.savefig(out/'figures'/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)
    ordered=sorted(selected,key=lambda r:r['path_length_units'])
    fig,ax=plt.subplots(figsize=(9,5.4));bars=ax.bar([label(r) for r in ordered],[r['path_length_units'] for r in ordered],color=[COLORS[r['curve_key']] for r in ordered],width=.62)
    ax.bar_label(bars,fmt='%.3f',padding=5);ax.set_ylim(0,max(r['path_length_units'] for r in selected)*1.18)
    ax.set_ylabel('경로 길이 L [unit]');ax.set_title('평가기준별 우승 후보 5개의 경로 길이',pad=15)
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
    fig.text(.5,.015,'동일 조건 · 유효 후보 12/42개 · 목적별 결정론적 우승 후보 5개를 표시',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.16);save(fig,'01_length_comparison')
    fig,ax=plt.subplots(figsize=(10,6.2))
    for family in dict.fromkeys(r['curve_key'] for r in valid):
        group=[r for r in valid if r['curve_key']==family]
        ax.scatter([r['path_length_units'] for r in group],[r['model_electrical_energy_j'] for r in group],c=COLORS[family],label=FAMILIES[family],s=65,alpha=.85)
    offsets={'hermite':(8,10),'arc':(45,34),'bezier':(-38,-30),'cycloid':(38,-7),'limacon':(-100,14)}
    for r in selected:
        ax.scatter(r['path_length_units'],r['model_electrical_energy_j'],facecolors='none',edgecolors='#172033',s=135,linewidths=1)
        ax.annotate(label(r),(r['path_length_units'],r['model_electrical_energy_j']),xytext=offsets[r['curve_key']],textcoords='offset points',fontsize=9,
                    arrowprops={'arrowstyle':'-','color':'#64748b','lw':.7},bbox={'fc':'white','ec':'none','alpha':.9,'pad':1.5})
    ax.set(xlabel='경로 길이 L [unit]',ylabel='모형 에너지 [J]',title='같은 상황에서 거리와 모형 에너지의 관계')
    ax.set_xlim(6.5,31);ax.set_ylim(60,315)
    ax.legend(fontsize=9,loc='upper left',bbox_to_anchor=(1.01,1));ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.015,'에너지 검증 통과 후보 12개 · 테두리 점은 목적별 우승 후보 · 실측 소비전력 아님',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.15,right=.80);save(fig,'02_distance_energy')
    fig,ax=plt.subplots(figsize=(8,7))
    for r in selected:
        g=geometry[geometry.run_id==r['run_id']]
        ax.plot(g.geometry_x,g.geometry_y,color=COLORS[r['curve_key']],label=label(r),lw=2)
    for o in obstacles:
        ax.add_patch(plt.Circle((o.center_x,o.center_y),o.radius_units,fc='#fbdede',ec='#d64545',lw=1.5))
        ax.annotate(o.obstacle_id,(o.center_x,o.center_y),ha='center',va='center',fontsize=9)
    # One explicit robot pose, not a swept-area or whole-path collision proof.
    state=j['samples'][-1];a,b=state['theta1_rad'],state['theta2_rad']
    elbow=(arm.base[0]+arm.link1*math.cos(a),arm.base[1]+arm.link1*math.sin(a))
    endpoint=(elbow[0]+arm.link2*math.cos(a+b),elbow[1]+arm.link2*math.sin(a+b))
    ax.plot([*arm.base[:1],elbow[0],endpoint[0]],[arm.base[1],elbow[1],endpoint[1]],'-o',c='#26334d',lw=2.5,label='리마송 도착 자세',zorder=4)
    ax.scatter(inp['start_x'],inp['start_y'],marker='s',s=65,c='#172033',zorder=5);ax.annotate('시작',(inp['start_x'],inp['start_y']),xytext=(7,8),textcoords='offset points')
    ax.scatter(inp['goal_x'],inp['goal_y'],marker='*',s=150,c='#f59e0b',zorder=5);ax.annotate('도착',(inp['goal_x'],inp['goal_y']),xytext=(-30,-18),textcoords='offset points')
    ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='x [unit]',ylabel='y [unit]',title='동일 조건에서 선택된 끝점 경로와 장애물')
    ax.grid(alpha=.16);ax.legend(fontsize=9,loc='upper left');fig.text(.5,.015,'곡선은 끝점 경로이며 링크 전체의 점유 영역은 아님 · 로봇팔은 리마송의 도착 자세만 표시',ha='center',fontsize=8)
    fig.subplots_adjust(bottom=.13);save(fig,'03_paths_and_obstacle')
    ranks={r['objective']:r for r in labels}
    h=byid[ranks['distance']['winner_run_id']];c=byid[ranks['model_energy']['winner_run_id']];a=byid[ranks['report_balanced']['winner_run_id']]
    l=byid[ranks['obstacle_penalty']['winner_run_id']]
    rows=['| 기준 | 선택 후보 | 평가값 | 동점 후보 수 |','|---|---|---:|---:|']
    for row in labels:
        rows.append(f"| {NAMES[row['objective']]} | {row['winner_run_id'].replace('interactive_','')} | {row['raw_metric_value']:.8g} | {int(row['tie_count'])} |")
    captions={
      '01_length_comparison':'동일 상황의 유효 후보 12개 중 9개 평가기준의 결정론적 우승 후보를 중복 제거한 5개를 비교했다. 곡선 종류 전체의 평균 비교가 아니며, 각 막대는 특정 굽힘값과 팔꿈치 해를 갖는 후보 하나다.',
      '02_distance_energy':'거리와 에너지 순위가 반드시 일치하지 않는다. 점 12개는 동일 상황의 후보로 서로 독립인 12개 환경 표본이 아니다. 모든 점은 selection_valid 및 energy_selection_ready를 통과했다. 에너지는 저장된 수직 평면·모터 모형의 추정치다.',
      '03_paths_and_obstacle':'목적별 우승 후보 5개의 끝점 경로를 동일 축척으로 그렸다. 링크 전체 충돌 판정은 저장된 경로 검사 결과를 따른다. 곡선이 장애물과 만나지 않는 그림만으로 로봇팔 전체의 무충돌을 판정하지 않는다.'}
    write(out/'figure_captions.csv',[{'figure':k,'caption_ko':v} for k,v in captions.items()])
    report=f'''# 분석자료 1: 단일 상황 비교

## 입력과 재현 확인

로봇 밑부분 (0, 0), 링크 길이 {inp['link1']} 및 {inp['link2']} unit,
시작 (2, 0), 도착 (-4.8, -1.5), 이동시간 3 s.
저장 JSON의 장애물은 중심 (-1.3, -2), 반지름 0.45 unit인 원 하나다.
화면의 장애물 이름 O4와 JSON의 O1은 표시 ID가 다르다. 분석은 JSON과 CSV의 수치에 근거한다.
좌표 1 unit={cfg.workspace_unit_m} m, 수직 평면. 나머지 물리 설정은 experiment_metadata.json 참조.

42개 후보를 저장 입력으로 전부 재계산하여 유효성·주요 지표·승자 라벨을 대조했다.
유효 후보 {len(valid)}/42개, 제외 후보 {42-len(valid)}/42개다.
원본 CSV의 후보마다 시간 및 기하 표본이 각각 161개이며 결합 키 중복이 없다.
사용자가 직접 입력한 사례로, train 데이터 전체에 대한 통계가 아니며 데이터 분할에 추가하지 않았다.

## 기준별 우승 후보

{chr(10).join(rows)}

종합 행의 값은 최소화하는 비용이다. 표시 점수는 100-비용={a['report_balanced_reward_0_100']:.8f}점이다.
9개 목적의 승자 라벨은 모두 ready다. 곡률은 2개 동점이며 저장된 결정 규칙의 후보만 대표로 표시했다.

## 관찰 결과

- 가장 짧은 유효 후보는 에르미트 아래 해: L={h['path_length_units']:.6f} unit, 모형 에너지={h['model_electrical_energy_j']:.6f} J.
- 모형 에너지 최소 후보는 사이클로이드 위 해: L={c['path_length_units']:.6f} unit, 에너지={c['model_electrical_energy_j']:.6f} J.
  이 후보는 최단 후보보다 경로가 {(c['path_length_units']/h['path_length_units']-1)*100:.2f}% 길지만, 모형 에너지는 {(1-c['model_electrical_energy_j']/h['model_electrical_energy_j'])*100:.2f}% 작았다.
- 종합점수의 우승 후보는 원호 위 해: L={a['path_length_units']:.6f} unit, 에너지={a['model_electrical_energy_j']:.6f} J, 점수={a['report_balanced_reward_0_100']:.6f}점.
- 화면에 적용한 리마송 아래 해는 접근 벌점 최소 후보: O={l['obstacle_penalty_s_per_unit2']:.6f} s/unit².
  하지만 경로 길이 {l['path_length_units']:.6f} unit, 모형 에너지 {l['model_electrical_energy_j']:.6f} J로 각각의 최소 후보는 아니다.
  이 리마송의 최소 여유는 {l['min_obstacle_clearance_units']:.6f} unit로, 최소 여유가 가장 큰 후보를 뜻하지 않는다.

## 해석 범위와 제한

각 결과는 준비된 42개 후보 안에서만 최적이다. 무한히 많은 곡선 중 전역 최적이나 특정 곡선 종류의 보편적 우월성을 뜻하지 않는다.
5개 비교 후보는 목적별 우승 후보의 합집합이다. 한 목적의 대표가 모든 목적에 공정한 대표라는 가정은 하지 않는다.
12개의 유효 후보를 모두 거리-에너지 그림에 표시했고, 제외 후보도 all_42_candidates.csv에 보존했다.
모형 에너지는 양의 기계적 일+동손 추정치이며 실측 소비전력이 아니다. 순 기계적 일과 혼동하지 않는다.
기존 관절 노력도 E_theta는 rad²/s이며 에너지 J가 아니다. 저크와 곡률도 서로 다른 지표다.
종합점수는 이 상황의 유효 후보 안에서 정규화한 값이다. 거리 .25, 모형 에너지 .20, 관절 저크 .20, 정칙화 곡률 .20, 접근 벌점 .15의 기존 가중치를 그대로 썼다.
가중치 민감도나 통계적 유의성은 검증하지 않았다. 641점 물리 적분값은 비교용 고해상도 기준이며 정확한 참값 보증이 아니다.
보통 곡률과 프로그램의 정칙화 곡률 비용을 혼동하지 않는다. 시간 표본 재적분은 headline 적응형/641점 값과 근사적으로만 일치할 수 있다.

## 그림 캡션

'''+ '\n\n'.join(f'### {k}\n\n{v}' for k,v in captions.items())
    (out/'분석설명.md').write_text(report,encoding='utf-8')
    formula=['# 팀원용 곡선식','표시 계수는 반올림이며 원본 식·정밀 파라미터는 selected_equations.csv와 입력 파일에 보존한다.',
             '공통: 0≤u≤1, u=q(t)=10(t/3)^3−15(t/3)^4+6(t/3)^5, 0≤t≤3 s. u는 진행률이며 φ는 각도다.']
    for r in selected:
        e=eq[r['run_id']];p=json.loads(r['curve_parameters_json']);key=r['curve_key']
        formula.append('\n## '+label(r)+' / '+short(r))
        if key=='arc':
            formula.append(f"α(u)={p['theta_start']:.12g}+({p['theta_delta']:.12g})u\nx(u)={p['center_x']:.12g}+{p['radius']:.12g}cos α(u)\ny(u)={p['center_y']:.12g}+{p['radius']:.12g}sin α(u)\nL=R|Δα|={p['radius']*abs(p['theta_delta']):.12g} unit（해석값）")
        elif key=='limacon':
            phi=p['phase'];rot=p['rotation'];s=p['scale'];side=p['side'];r0=p['a']+p['b']*math.cos(phi)
            x0=r0*math.cos(phi);y0=side*r0*math.sin(phi)
            cx=inp['start_x']-s*(math.cos(rot)*x0-math.sin(rot)*y0);cy=inp['start_y']-s*(math.sin(rot)*x0+math.cos(rot)*y0)
            beta=math.atan2(math.sin(rot),math.cos(rot))
            formula.append(f"φ(u)={phi:.12g}+{p['sweep']:.12g}u\nr(φ)=1+0.79cosφ\nx(φ)={cx:.12g}+{s:.12g}r(φ)cos({beta:.12g}+φ)\ny(φ)={cy:.12g}+{s:.12g}r(φ)sin({beta:.12g}+φ)\nL={s:.12g}∫[−3π/5,3π/5] √(1.6241+1.58cosφ) dφ（数値積分）")
        elif key=='cycloid':
            d=math.hypot(inp['goal_x']-inp['start_x'],inp['goal_y']-inp['start_y']);ex=(inp['goal_x']-inp['start_x'])/d;ey=(inp['goal_y']-inp['start_y'])/d
            formula.append(str(e['native_equation'])+f"\nx(φ)=2+({ex:.12g})X_local+({-ey:.12g})Y_local\ny(φ)=({ey:.12g})X_local+({ex:.12g})Y_local\n0≤φ≤2π. A와 B가 다른 일반화된 사이클로이드이며 표준 A=B의 길이 8A를 그대로 쓰면 안 된다.")
        else:
            curve=make_curve(key,(inp['start_x'],inp['start_y']),(inp['goal_x'],inp['goal_y']),shape=r['shape'])
            for axis,coeff in zip(('x','y'),polynomial_coefficients(curve)):
                terms=[]
                for power,value in enumerate(coeff):
                    if abs(value)<1e-12: continue
                    terms.append(f"{value:+.12g}"+('' if power==0 else 'u' if power==1 else f'u^{power}'))
                formula.append(f"{axis}(u)="+''.join(terms).lstrip('+'))
            if key=='bezier':formula.append('이 후보는 대칭 제어점 때문에 3차항이 소거되어 2차식이 된다. 1e-12 미만의 부동소수점 잔차는 표시에서 생략했다.')
        formula.extend(['관절각 합성식:',str(e['ik_cosine_equation']),str(e['theta2_t_equation']),str(e['theta1_t_equation']),str(e['branch_policy'])])
    (out/'팀원용_곡선식.md').write_text('\n\n'.join(formula).replace('数値積分','수치적분').replace('（','(').replace('）',')'),encoding='utf-8')
    summary_fields=['후보','선정 기준','길이 [unit]','모형 에너지 [J]','관절 노력도 [rad²/s]','저크 비용 [rad²/s⁵]','Kε [1/unit]','접근 벌점 [s/unit²]','종합점수 [점]']
    summary_rows=[]
    for r in selected:
        reasons='/'.join(NAMES[x['objective']] for x in labels if x['winner_run_id']==r['run_id'])
        summary_rows.append([label(r),reasons,r['path_length_units'],r['model_electrical_energy_j'],r['joint_velocity_effort'],r['joint_jerk_effort'],r['regularized_curvature_energy_inv_unit'],r['obstacle_penalty_s_per_unit2'],r['report_balanced_reward_0_100']])
    settings=[['기준', '입력값'],*[[k,str(v)] for k,v in inp.items()],*[[k,str(v)] for k,v in j['physics_model'].items()],['장애물',json.dumps(j['obstacles'],ensure_ascii=False)],['출처','분석자료_1/robot_path_run.json 및 robot_path_comparison_20261021_102523_870740/*.csv']]
    payload=dict(summary_headers=summary_fields,summary_rows=summary_rows,all_headers=FIELDS,all_rows=[[r[k] for k in FIELDS] for r in records],
                 winner_headers=['기준','결정 후보','값','동점 수','상태'],winner_rows=[[NAMES[r['objective']],r['winner_run_id'].replace('interactive_',''),r['raw_metric_value'],r['tie_count'],r['label_status']] for r in labels],settings=settings,
                 selected_labels=[label(r) for r in selected],colors=[COLORS[r['curve_key']] for r in selected],counts={'all':42,'valid':len(valid)},
                 valid_rows=[[label(r)+' '+short(r),r['path_length_units'],r['model_electrical_energy_j']] for r in valid])
    (out/'workbook_payload.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    checks=dict(status='passed',candidate_count=42,valid_candidates=len(valid),selected_candidates=len(selected),
                full_42_replay=True,winner_replay=True,unique_sample_keys=True,samples_per_run=161,
                source_hashes=hashes,comparison_method='same exported scenario; no held-out data',
                numerical_replay_tolerance={'rtol':2e-10,'atol':1e-9},software={'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__})
    assert hashes=={str(p.relative_to(src)):sha(p) for p in source_files}
    (out/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'experiment_metadata.json').write_text(json.dumps({'inputs':inp,'obstacles':j['obstacles'],'physics_model':j['physics_model'],'source_hashes':hashes,'selected_run_ids':selected_ids},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'valid':len(valid),'selected':selected_ids,'replay':'passed'},ensure_ascii=False))

if __name__=='__main__':main()
