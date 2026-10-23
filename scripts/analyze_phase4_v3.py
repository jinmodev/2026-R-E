"""Train-only, versioned R&E descriptive analysis; never trains a model."""
from __future__ import annotations
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from robot_planner.release_io import iter_release_metrics
from robot_planner.phase4 import RAW_OBJECTIVES, _winner_key

DATA = ROOT / 'data/phase4_v3_release'
OUT = ROOT / 'outputs/phase4_v3_analysis'
SEED = 20260913
FAMILIES = ['line', 'arc', 'bezier', 'hermite', 'cycloid', 'limacon']
NAMES = dict(zip(FAMILIES, ['직선', '원호', '베지어', '에르미트', '사이클로이드', '리마송']))
COLORS = dict(zip(FAMILIES, ['#3264D8', '#D78222', '#259067', '#CB5367', '#8864BB', '#189DAA']))
OBJECTIVES = {**RAW_OBJECTIVES, 'balanced': 'balanced_cost_0_100'}
LABELS = dict(zip(OBJECTIVES, ['거리 L', '관절 노력도 Eθ', '각저크 J', '정칙화 곡률 Kε', '접근 벌점 O', '종합 비용']))
FIELDS = {'L': OBJECTIVES['distance'], 'E_theta': OBJECTIVES['joint_effort'],
          'A': 'joint_acceleration_effort', 'J': OBJECTIVES['jerk'],
          'K_epsilon': OBJECTIVES['curvature'], 'O': OBJECTIVES['safety']}
UNITS = dict(zip(OBJECTIVES, ['unit', 'rad²/s', 'rad²/s⁵', '1/unit', 's/unit²', '0–100']))
CONDITIONS = ['scenario_type', 'obstacle_layout', 'obstacle_count', 'ratio_bin',
              'motion_bin', 'angle_bin', 'start_radius_bin', 'goal_radius_bin']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(name, rows):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
    return df


def hashed(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()


def partition(sid):
    p = DATA / 'partitions/train' / sid
    receipt = json.loads((p / 'done.json').read_text(encoding='utf-8'))
    assert receipt['data_split'] == 'train'
    return p / receipt['attempt']


def candidate_winner(rows, field):
    best = min(float(r[field]) for r in rows)
    tied = [r for r in rows if math.isclose(float(r[field]), best, rel_tol=1e-12, abs_tol=1e-12)]
    return min(tied, key=lambda r: _winner_key(r, field)), tied


def classify(rows):
    """Exclusive reporting groups without interpreting unknown as false."""
    valid = sum(r['selection_valid'] for r in rows)
    unresolved = sum(r['candidate_status'] == 'unresolved' for r in rows)
    u = sum(r['has_unreachable_witness'] for r in rows)
    c = sum(r['has_collision_witness'] for r in rows)
    if valid: reason = 'has_valid'
    elif unresolved: reason = 'unresolved_or_calculation'
    elif u == len(rows): reason = 'all_unreachable_witness'
    elif c == len(rows) and not u: reason = 'all_collision_witness'
    elif u and c: reason = 'mixed_unreachable_collision'
    else: reason = 'other_calculation_or_schema'
    cohort = 'excluded_unresolved' if unresolved else ('main' if valid else 'excluded_no_valid')
    return dict(accepted=valid, unresolved=unresolved, unreachable_witness=u,
                collision_witness=c, reason=reason, cohort=cohort)


def add_bins(s):
    s = s.copy()
    s['link1_over_link2'] = s.link1 / s.link2
    for name, values, edges in [
        ('ratio_bin', s.link1_over_link2, [-np.inf, .75, 4/3, np.inf]),
        ('motion_bin', s.chord_length / (s.link1+s.link2), [0, .5, 1, 2.01]),
        ('angle_bin', s.angular_separation_deg, [0, 60, 120, 181]),
        ('start_radius_bin', s.start_radius / (s.link1+s.link2), [0, .4, .8, 1.01]),
        ('goal_radius_bin', s.goal_radius / (s.link1+s.link2), [0, .4, .8, 1.01])]:
        b = pd.cut(values, edges, right=False)
        if b.isna().any(): raise ValueError(f'Out-of-range condition: {name}')
        s[name] = b.astype(str)
    return s


def preserved_hashes():
    paths = []
    for folder in ['outputs/phase4_analysis', 'outputs/phase4_quality', 'outputs/phase4_v3_release_quality',
                   'outputs/phase4_v3_quality', 'src/robot_planner']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def load():
    rows = []; geom = []; altered_names = 0
    for r in iter_release_metrics(DATA):
        if r['curve_name_ko'] != NAMES[r['curve_key']]: altered_names += 1
        # Long formulas/arrays remain in their source; scalar analysis frame only.
        d = {k:v for k,v in r.items() if not isinstance(v, (list,dict)) and 'equation' not in k
             and k not in ('curve_parameters_json', 'obstacles_json')}
        d['has_unreachable_witness'] = bool(r['path_check']['unreachable_witnesses'])
        d['has_collision_witness'] = bool(r['path_check']['collision_witnesses'])
        d['curve_name_ko'] = NAMES[r['curve_key']]
        rows.append(d)
        for field, check in r['geometry_checks'].items():
            vals = check['values_by_n']; ref = vals['641']
            geom.append(dict(scenario_id=r['scenario_id'], run_id=r['run_id'], field=field,
                candidate_status=r['candidate_status'], value161=vals['161'], value321=vals['321'],
                value641=ref, absolute_difference=abs(vals['161']-ref),
                relative_difference=abs(vals['161']-ref)/abs(ref) if ref else None,
                tolerance_ratio=abs(vals['161']-ref)/(check['atol']+check['rtol']*abs(ref)),
                passed=check['passed']))
    m = pd.DataFrame(rows)
    assert set(m.data_split) == {'train'} and len(m) == 126000 and m.run_id.is_unique
    assert m.groupby('scenario_id').size().eq(42).all()
    assert m.selection_valid.eq(m.candidate_status.eq('accepted')).all()
    assert np.isfinite(m.loc[m.selection_valid, list(FIELDS.values())].to_numpy(float)).all()
    s = m.drop_duplicates('scenario_id')[['scenario_id','scenario_type','obstacle_layout','obstacle_count',
        'link1','link2','link_ratio','chord_length','angular_separation_deg','start_radius','goal_radius', 'duration_s']]
    s = add_bins(s)
    exclusions = pd.DataFrame([dict(scenario_id=sid, **classify(g.to_dict('records')))
                              for sid,g in m.groupby('scenario_id', sort=True)])
    s = s.merge(exclusions, on='scenario_id', validate='one_to_one')
    m = m.merge(s[['scenario_id','cohort']], on='scenario_id', validate='many_to_one')
    table('scenario_exclusions', s)
    table('no_valid_summary', s.groupby(['reason','cohort']).size().reset_index(name='scenarios'))
    table('analysis_flow', [dict(scope='composition_and_validity', included=len(s), excluded=0),
        dict(scope='main_winner_paired_relationship', included=int(s.cohort.eq('main').sum()), excluded=int(s.cohort.ne('main').sum())),
        dict(scope='excluded_unresolved', included=int(s.cohort.eq('excluded_unresolved').sum()), excluded=0),
        dict(scope='excluded_no_valid', included=int(s.cohort.eq('excluded_no_valid').sum()), excluded=0)])
    table('geometry_convergence', geom)
    m.to_parquet(OUT/'train_analysis_metrics.parquet', index=False)
    print(f'Loaded train: {len(s)} scenarios, {len(m)} candidates. Cohorts: {s.cohort.value_counts().to_dict()}', flush=True)
    return m, s, altered_names


def summarize(m, s):
    composition=[]; conditional=[]; exclusions=[]
    any_valid=m.groupby(['scenario_id','curve_key']).selection_valid.any().reset_index()
    for col in CONDITIONS:
        for b,g in s.groupby(col, observed=True, sort=True):
            n=len(g)
            composition.append(dict(variable=col, category=str(b), scenarios=n, denominator=len(s), fraction=n/len(s), small_n=n<30))
            for cohort in ['main','excluded_no_valid','excluded_unresolved']:
                exclusions.append(dict(condition=col, bin=str(b), cohort=cohort, scenarios=int(g.cohort.eq(cohort).sum()), denominator=n))
            z=any_valid[any_valid.scenario_id.isin(g.scenario_id)]
            for k in FAMILIES:
                good=int(z.loc[z.curve_key.eq(k),'selection_valid'].sum())
                conditional.append(dict(condition=col, bin=str(b), curve_key=k, scenarios=n, scenarios_any_valid=good,
                    scenario_valid_rate=good/n, small_n=n<30))
    table('composition', composition);table('conditional_validity',conditional);table('conditional_exclusions',exclusions)
    table('geometry_summary',s[['link1','link2','link_ratio','chord_length','start_radius','goal_radius','angular_separation_deg','duration_s']].describe().reset_index(names='statistic'))
    vals=[]
    for k in FAMILIES:
        g=m[m.curve_key.eq(k)];reach=g.fully_reachable.eq(True)
        nr=int(reach.sum()); nc=int((reach&g.collision_free.eq(True)).sum()); nv=int(g.selection_valid.sum())
        ns=int(g.groupby('scenario_id').selection_valid.any().sum())
        vals.append(dict(curve_key=k,curve_name=NAMES[k],variants_per_scenario=len(g)//len(s),candidates=len(g),
            reachable=nr,reach_unknown=int(g.fully_reachable.isna().sum()),collision_free_among_reachable=nc,
            collision_unknown_among_reachable=int((reach&g.collision_free.isna()).sum()), valid=nv,
            unresolved=int(g.candidate_status.eq('unresolved').sum()),scenarios=len(s),scenarios_any_valid=ns,
            reachable_rate=nr/len(g),conditional_collision_free_rate=nc/nr if nr else None,
            valid_rate=nv/len(g),scenario_valid_rate=ns/len(s)))
    table('validity',vals)
    v=m[m.selection_valid & m.cohort.eq('main')]
    winners=[]
    for sid,g in v.groupby('scenario_id',sort=True):
        rr=g.to_dict('records')
        for obj,field in OBJECTIVES.items():
            w,tied=candidate_winner(rr,field)
            winners.append(dict(scenario_id=sid,objective=obj,winner_run_id=w['run_id'],curve_key=w['curve_key'],
                value=w[field],tie_count=len(tied),tied_run_ids_json=json.dumps(sorted(r['run_id'] for r in tied)),
                obstacle_count=w['obstacle_count']))
    w=table('main_winners',winners); family=m.set_index('run_id').curve_key.to_dict();shares=[]
    for scope, ids in [('main',set(v.scenario_id)), ('obstacles_only',set(s.loc[s.cohort.eq('main')&s.obstacle_count.gt(0),'scenario_id']))]:
        for obj in OBJECTIVES:
            g=w[w.objective.eq(obj)&w.scenario_id.isin(ids)];n=len(g);fraction={k:0. for k in FAMILIES}
            for ties in g.tied_run_ids_json.map(json.loads):
                for rid in ties: fraction[family[rid]]+=1/len(ties)
            for k in FAMILIES:
                wins=int(g.curve_key.eq(k).sum())
                shares.append(dict(scope=scope,objective=obj,curve_key=k,deterministic_wins=wins,
                    fractional_wins=fraction[k],denominator_scenarios=n,tied_scenarios=int(g.tie_count.gt(1).sum()),
                    deterministic_rate=wins/n if n else None,fractional_rate=fraction[k]/n if n else None))
    table('winner_shares',shares)
    paired=[]
    for obj,field in OBJECTIVES.items():
        best=v.groupby(['scenario_id','curve_key'])[field].min().unstack('curve_key').reindex(columns=FAMILIES)
        for a,b in itertools.combinations(FAMILIES,2):
            both=best[[a,b]].dropna();diff=both[a]-both[b]
            tie=np.isclose(both[a],both[b],rtol=1e-12,atol=1e-12)
            paired.append(dict(objective=obj,unit=UNITS[obj],family_a=a,family_b=b,paired_scenarios=len(both),
                median_a_minus_b=diff.median() if len(both) else None,
                a_lower_fraction=float(((diff<0)&~tie).mean()) if len(both) else None,
                tie_fraction=float(tie.mean()) if len(both) else None,small_n=len(both)<30))
    table('paired_objective_comparisons',paired)
    reps=v.assign(hash=v.run_id.map(hashed)).sort_values('hash').drop_duplicates('scenario_id').drop(columns='hash')
    table('relationship_representatives',reps[['scenario_id','run_id','curve_key',*FIELDS.values(),
        'min_obstacle_clearance_units','duration_s','link1','link2',*['normalized_'+o for o in RAW_OBJECTIVES]]])
    relationships=[]
    for scope,xx,yy in [('raw',FIELDS['L'],FIELDS['E_theta']),('raw',FIELDS['J'],FIELDS['K_epsilon']),
        ('raw','min_obstacle_clearance_units',FIELDS['O']),('within_scenario','normalized_distance','normalized_joint_effort'),
        ('within_scenario','normalized_jerk','normalized_curvature')]:
        z=reps[[xx,yy]].dropna()
        rho=z[xx].rank().corr(z[yy].rank()) if len(z)>1 and z[xx].nunique()>1 and z[yy].nunique()>1 else None
        relationships.append(dict(scope=scope,x=xx,y=yy,scenarios=len(z),spearman_rho=rho))
    table('relationships',relationships)
    # Stored label reproduction is independent of the analysis's tie aggregation.
    checked=0
    for sid,g in w.groupby('scenario_id'):
        original=json.loads((partition(sid)/'scenario_winners.json').read_text(encoding='utf-8'))
        orig={r['objective']:r for r in original}
        for r in g.to_dict('records'):
            assert orig[r['objective']]['winner_run_id']==r['winner_run_id']
            assert json.loads(orig[r['objective']]['tied_run_ids_json'])==json.loads(r['tied_run_ids_json'])
            checked+=1
    return w,reps,checked


def plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,
        'svg.fonttype':'path','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'bold','figure.facecolor':'white'})
    return plt


def figures(m,s,w,reps):
    plt=plot_setup();caps=[]
    def save(fig,name,caption):
        from matplotlib.ticker import FuncFormatter
        for ax in fig.axes:
            if ax.get_xscale()=='symlog':ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
            if ax.get_yscale()=='symlog':ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
        fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
        caps.append(dict(figure=name,caption_ko=caption));print('Figure:',name,flush=True)
    f,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for a,col,label in zip(axes,['link1','link_ratio','obstacle_count'],['첫째 링크 길이 (unit)','짧은 링크 / 긴 링크','장애물 수 (개)']):
        a.hist(s[col],bins=18 if col!='obstacle_count' else np.arange(s[col].max()+2)-.5,color='#3264D8',edgecolor='white')
        a.set(xlabel=label,ylabel='시나리오 수')
    f.suptitle(f'Train 데이터 구성 (n={len(s):,} 상황)')
    save(f,'01_composition',f'train {len(s):,}개 시나리오. 후보를 독립 표본으로 세지 않았다. link_ratio는 짧은 링크/긴 링크다.')
    v=pd.read_csv(OUT/'validity.csv').set_index('curve_key').loc[FAMILIES];x=np.arange(6)
    f,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for a,col,title in zip(axes.flat,['reachable_rate','conditional_collision_free_rate','valid_rate','scenario_valid_rate'],
        ['도달 확인 / 전체 후보','무충돌 확인 / 도달 확인 후보','유효 / 전체 후보',f'유효 후보 ≥1 / {len(s):,} 상황']):
        bars=a.bar(x,v[col]*100,color=[COLORS[k] for k in FAMILIES]);a.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
        a.set(xticks=x,xticklabels=[NAMES[k] for k in FAMILIES],ylim=(0,110),ylabel='비율 (%)',title=title)
    save(f,'02_validity','후보 분모: 직선 6,000, 나머지 각 24,000. 조건부 무충돌 분모는 validity.csv의 reachable. 미확정은 성공으로 세지 않으며 별도 열에 기록했다. 같은 종류는 모든 그림에서 같은 색이다.')
    wins=pd.read_csv(OUT/'winner_shares.csv')
    for scope,name in [('main','03_winners'),('obstacles_only','04_obstacle_winners')]:
        f,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        for a,obj in zip(axes.flat,OBJECTIVES):
            z=wins[wins.scope.eq(scope)&wins.objective.eq(obj)].set_index('curve_key').loc[FAMILIES]
            a.bar(x-.18,z.deterministic_rate*100,.36,color=[COLORS[k] for k in FAMILIES],label='결정적 선택')
            a.bar(x+.18,z.fractional_rate*100,.36,color=[COLORS[k] for k in FAMILIES],alpha=.4,hatch='//',label='동점 1/k 배분')
            a.set(xticks=x,xticklabels=[NAMES[k] for k in FAMILIES],ylabel='승자 비율 (%)',title=f'{LABELS[obj]} (n={int(z.denominator_scenarios.iloc[0]):,})')
            a.tick_params(axis='x',labelsize=8)
        axes.flat[0].legend(fontsize=8)
        f.suptitle('검사 통과 후보 중 목적별 선택'+(' — 장애물이 있는 상황만' if scope!='main' else ' — 미확정 대안이 있는 상황 제외'))
        save(f,name,'주 비교 집합'+(' 중 장애물 있음' if scope!='main' else '')+'. 결정적 선택과 동점 후보별 1/k 배분을 분리했다. 직선은 2개, 나머지는 각 8개 후보라 탐색 기회가 다르다. 무장애물 O=0 동점은 안전성 우월이 아니다. 수치 허용오차 이내의 순위 안정까지 보증하지 않는다.')
    pairs=[(FIELDS['L'],FIELDS['E_theta'],'거리 L (unit)','관절 노력도 Eθ (rad²/s)'),
           (FIELDS['J'],FIELDS['K_epsilon'],'각저크 J (rad²/s⁵)','정칙화 곡률 Kε (1/unit)'),
           ('min_obstacle_clearance_units',FIELDS['O'],'표본 최소 링크 여유 (unit)','접근 벌점 O (s/unit²)')]
    f,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for a,(xx,yy,xl,yl) in zip(axes,pairs):
        z=reps[[xx,yy,'curve_key']].dropna()
        for k,g in z.groupby('curve_key'):a.scatter(g[xx],g[yy],s=10,alpha=.45,color=COLORS[k],label=NAMES[k])
        a.set(xlabel=xl,ylabel=yl,title=f'상황당 1개 후보 (n={len(z):,})');a.set_yscale('symlog',linthresh=.01)
        if xx==FIELDS['J']:
            a.set_xscale('symlog',linthresh=.01)
            a.set_xticks([1,100,10000,1e6,1e8,1e10])
    axes[0].legend(fontsize=8)
    save(f,'05_relationships','목적값과 독립적인 고정 SHA256 순서로 상황당 후보 하나를 선택했다. 세로축과 저크 가로축은 symlog, 선형 경계 0.01. 원 지표이며 인과효과가 아니다. 여유는 표본 최솟값이지 연속 최소의 정확한 참값이 아니다.')
    f,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for a,xx,yy in zip(axes,['normalized_distance','normalized_jerk'],['normalized_joint_effort','normalized_curvature']):
        for k,g in reps.groupby('curve_key'):a.scatter(g[xx],g[yy],color=COLORS[k],s=9,alpha=.4,label=NAMES[k])
        a.set(xlabel=xx+' (무차원)',ylabel=yy+' (무차원)',title=f'시나리오 내부 min–max (n={len(reps):,})')
    axes[0].legend(fontsize=8)
    save(f,'06_normalized_relationships','각 상황의 유효 후보 범위를 기준으로 한 정규화 관계다. 서로 다른 상황의 같은 정규화 값이 같은 원 비용을 뜻하지 않는다.')
    c=pd.read_csv(OUT/'conditional_validity.csv');z=c[c.condition.eq('obstacle_layout')]
    f,a=plt.subplots(figsize=(10,5.5),layout='constrained');pivot=z.pivot(index='bin',columns='curve_key',values='scenario_valid_rate')[FAMILIES]
    im=a.imshow(pivot,aspect='auto',vmin=0,vmax=1,cmap='Blues');ns=z.groupby('bin').scenarios.first()
    a.set(xticks=x,xticklabels=[NAMES[k] for k in FAMILIES],yticks=np.arange(len(pivot)),
        yticklabels=[f'{b} (n={ns[b]})' for b in pivot.index],title='장애물 배치별 유효 후보 확보율')
    for i in range(len(pivot)):
        for j in range(6):a.text(j,i,f'{pivot.iloc[i,j]:.1%}',ha='center',va='center',color='white' if pivot.iloc[i,j]>.6 else '#20314D')
    f.colorbar(im,ax=a,label='유효 후보 ≥1 상황 / 해당 배치 상황')
    save(f,'07_obstacle_conditions','배치별 전체 상황을 분모로 한 종류별 유효 후보 확보율. 막대 색과 달리 heatmap의 색은 비율을 뜻한다. n<30 표시는 CSV에 있다.')
    f,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for a,condition,title in zip(axes.flat,['ratio_bin','motion_bin','start_radius_bin','goal_radius_bin'],['L1/L2','직선거리 / 총길이','시작 반경 / 총길이','도착 반경 / 총길이']):
        z=c[c.condition.eq(condition)];bins=sorted(z.bin.unique());ns=z.groupby('bin').scenarios.first()
        for k in FAMILIES:
            g=z[z.curve_key.eq(k)].set_index('bin').loc[bins];a.plot(range(len(bins)),g.scenario_valid_rate*100,'o-',ms=4,color=COLORS[k],label=NAMES[k])
        a.set(xticks=range(len(bins)),xticklabels=[f'{b}\nn={ns[b]}' for b in bins],ylabel='유효 후보 확보율 (%)',ylim=(0,105),title=title)
    axes.flat[0].legend(fontsize=8,ncol=2)
    save(f,'08_geometry_conditions','고정 구간은 왼쪽 포함·오른쪽 제외. 각 종류의 유효 후보 ≥1 상황 수 / 해당 구간 전체 상황 수다. 조건별 인과효과를 식별한 분석이 아니다.')
    return save,caps


def load_case(rid):
    sid=rid.split('_')[0];folder=partition(sid)
    with gzip.open(folder/'trajectory_metrics.jsonl.gz','rt',encoding='utf-8') as f:
        r=next(json.loads(line) for line in f if json.loads(line)['run_id']==rid)
    cols=['run_id','sample_index','time_s','x','y','theta1_rad','theta2_rad','reachable_at_sample','geometry_u',
          'path_speed_units_per_u','regularized_curvature_integrand_inv_unit','obstacle_penalty_integrand_per_unit2']
    fr=pq.read_table(folder/'trajectory_samples.parquet',columns=cols,filters=[('run_id','=',rid)]).to_pandas().sort_values('sample_index')
    panels=[]
    with gzip.open(folder/'adaptive_panels.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            p=json.loads(line)
            if p['run_id']==rid:panels.append(p)
    return r,fr,panels


def draw_path(a,r,fr,plt,poses=True):
    k=r['curve_key'];a.plot(fr.x,fr.y,color=COLORS[k],lw=2,label=NAMES[k])
    for o in json.loads(r['obstacles_json']):a.add_patch(plt.Circle((o['center_x'],o['center_y']),o['radius_units'],color='#CB5367',alpha=.35))
    bx,by=r['base_x'],r['base_y'];total=r['link1']+r['link2']
    a.add_patch(plt.Circle((bx,by),total,fill=False,ls='--',lw=.7,color='#A0ABBA'))
    for tag,xx,yy in [('시작',r['start_x'],r['start_y']),('도착',r['goal_x'],r['goal_y'])]:
        a.scatter(xx,yy,s=24,color='#20314D');a.annotate(tag,(xx,yy),xytext=(5,5),textcoords='offset points',fontsize=8)
    if poses:
        for i in [0,40,80,120,160]:
            z=fr.iloc[i]
            if pd.notna(z.theta1_rad):
                jx=bx+r['link1']*math.cos(z.theta1_rad);jy=by+r['link1']*math.sin(z.theta1_rad)
                a.plot([bx,jx,z.x],[by,jy,z.y],'-o',color='#20314D',alpha=.28,ms=3)
    bad=fr[fr.reachable_at_sample.eq(False)]
    if len(bad):a.scatter(bad.x,bad.y,s=10,color='#CB5367',label='도달 불가 표본')
    if r['candidate_status']=='invalid':
        from robot_planner.curves import make_curve
        from robot_planner.kinematics import PlanarArm2R
        from robot_planner.phase4_evaluation import _curve_value_derivatives
        curve=make_curve(r['curve_key'],(r['start_x'],r['start_y']),(r['goal_x'],r['goal_y']),shape=r['shape'],samples=161)
        arm=PlanarArm2R(r['link1'],r['link2'],base=(bx,by))
        if r['path_check']['unreachable_witnesses']:
            u=r['path_check']['unreachable_witnesses'][0];xx,yy,*_=_curve_value_derivatives(curve,u)
            a.scatter(xx,yy,marker='x',s=70,color='#B32D40',label=f'도달 위반 u={u:.3f}')
        if r['path_check']['collision_witnesses']:
            u=r['path_check']['collision_witnesses'][0];xx,yy,*_=_curve_value_derivatives(curve,u)
            q=arm.inverse((xx,yy),r['elbow_mode'])
            jx=bx+r['link1']*math.cos(q.theta1);jy=by+r['link1']*math.sin(q.theta1)
            a.plot([bx,jx,xx],[by,jy,yy],'-o',lw=2,color='#B32D40',ms=3,label=f'충돌 자세 u={u:.3f}')
        a.legend(fontsize=7,loc='best')
    a.set(aspect='equal',xlabel='x (unit)',ylabel='y (unit)')


def cases(m,w,save):
    plt=plot_setup();selection=[];records=[];checks=[];node_rows=[]
    valid=m[m.selection_valid&m.cohort.eq('main')]
    for k in FAMILIES:
        g=valid[valid.curve_key.eq(k)]
        if len(g):selection.append((min(g.run_id,key=hashed),'family_'+k,'주 비교 집합에서 종류별 고정 해시 선정'))
    for column,tag in [('has_unreachable_witness','unreachable'),('has_collision_witness','collision')]:
        g=m[m[column]]
        if len(g):selection.append((min(g.run_id,key=hashed),tag,'적응형 경로 검사 위반 witness가 있는 고정 해시 사례'))
    for rid,tag,reason in selection:
        r,fr,panels=load_case(rid)
        eq={k:v for k,v in r.items() if 'equation' in k or k in ('curve_parameters_json','branch_policy','unwrap_policy')}
        records.append(dict(scenario_id=r['scenario_id'],run_id=rid,role=tag,reason=reason,
            candidate_status=r['candidate_status'],equations=eq,metrics={k:r[v] for k,v in FIELDS.items()},
            input={k:r[k] for k in ['base_x','base_y','link1','link2','start_x','start_y','goal_x','goal_y','duration_s','shape','elbow_mode','obstacles_json']},
            path_check=r['path_check'],source_partition=str(partition(r['scenario_id']).relative_to(ROOT))))
        fr.to_csv(OUT/f'case_{tag}_samples.csv',index=False,encoding='utf-8-sig')
        f,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        draw_path(axes[0],r,fr,plt);axes[0].set_title('경로·장애물·5개 시각의 로봇팔')
        axes[1].plot(fr.time_s,fr.theta1_rad,label='θ1(t)',color=COLORS[r['curve_key']]);axes[1].plot(fr.time_s,fr.theta2_rad,'--',label='θ2(t)',color=COLORS[r['curve_key']])
        axes[1].set(xlabel='시간 t (s)',ylabel='관절각 (rad)',title='역기구학 합성식의 시간 표본');axes[1].legend()
        f.suptitle(f'{rid} — {r["candidate_status"]}',fontsize=11)
        save(f,'case_'+tag,f'{reason}. scenario_id={r["scenario_id"]}, run_id={rid}. 식은 representative_equations.json. 빈 관절각은 계산되지 않은 값이다. invalid 사례에 시간 표본이 정상처럼 보여도 적응형 witness로 제외될 수 있다.')
        if r['selection_valid']:
            f,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
            for a,key in zip(axes.flat,FIELDS):
                field=FIELDS[key]
                if key in ('E_theta','A','J'):
                    ps=[p for p in panels if p['metric']==field];ts=[];ys=[];sums=[]
                    for p in ps:
                        ts.extend(p['time_s']);ys.extend(p['integrand']);sums.extend(ww*yy for ww,yy in zip(p['weights_s'],p['integrand']))
                        node_rows.extend(dict(run_id=rid,metric=key,time_s=tt,weight_s=ww,integrand=yy) for tt,ww,yy in zip(p['time_s'],p['weights_s'],p['integrand']))
                    order=np.argsort(ts);a.plot(np.array(ts)[order],np.array(ys)[order],color=COLORS[r['curve_key']]);a.set_xlabel('시간 t (s)')
                    replay=math.fsum(sums);assert math.isclose(replay,r[field],rel_tol=1e-12,abs_tol=1e-9)
                    a.set_ylabel({'E_theta':'rad²/s²','A':'rad²/s⁴','J':'rad²/s⁶'}[key])
                    checks.append(dict(run_id=rid,metric=key,replayed=replay,summary=r[field],absolute_error=abs(replay-r[field])))
                else:
                    xx,yy=('geometry_u','path_speed_units_per_u') if key=='L' else (('geometry_u','regularized_curvature_integrand_inv_unit') if key=='K_epsilon' else ('time_s','obstacle_penalty_integrand_per_unit2'))
                    a.plot(fr[xx],fr[yy],color=COLORS[r['curve_key']]);a.set_xlabel('시간 t (s)' if key=='O' else '기하 매개변수 u')
                    a.set_ylabel({'L':'unit','K_epsilon':'1/unit','O':'1/unit²'}[key]);replay=float(np.trapezoid(fr[yy],fr[xx]))
                    assert math.isclose(replay,r[field],rel_tol=1e-10,abs_tol=1e-8)
                    checks.append(dict(run_id=rid,metric=key,replayed=replay,summary=r[field],absolute_error=abs(replay-r[field])))
                a.set_title(f'{key}: 적분값 {r[field]:.5g}')
                peak=max(float(np.nanmax(line.get_ydata())) for line in a.lines)
                a.set_ylim(0,1.08*peak if peak>0 else 1)
            f.suptitle(rid+' — 피적분함수 (패널마다 단위가 다름)',fontsize=11)
            save(f,'integrands_'+tag,f'{rid}. Eθ/A/J는 저장된 적응형 노드에서 그렸으며 적분값은 가중합이다. L/Kε/O는 161점 적분을 재현했다. 피적분함수의 세로축 단위가 다르므로 높이를 직접 비교하지 않는다.')
    # An observed conflict must have disjoint tie sets, not merely different deterministic IDs.
    trades=[];indexed=m.set_index('run_id')
    for sid,g in w[w.objective.isin(['distance','joint_effort','jerk'])].groupby('scenario_id'):
        rr=g.to_dict('records')
        for a,b in itertools.combinations(rr,2):
            if not set(json.loads(a['tied_run_ids_json']))&set(json.loads(b['tied_run_ids_json'])):
                ma=indexed.loc[a['winner_run_id']];mb=indexed.loc[b['winner_run_id']]
                if (ma.curve_key,ma['shape'])!=(mb.curve_key,mb['shape']):trades.append((sid,a,b));break
    trade_json=[]
    if trades:
        sid,a,b=min(trades,key=lambda z:hashed(z[0]));f,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
        for ax,z in zip(axes,[a,b]):
            r,fr,_=load_case(z['winner_run_id']);draw_path(ax,r,fr,plt)
            ax.set_title(f'{LABELS[z["objective"]]} 선택\n{r["run_id"]}',fontsize=9)
            trade_json.append(dict(objective=z['objective'],run_id=r['run_id'],scenario_id=sid,
                equations={k:v for k,v in r.items() if 'equation' in k},metrics={o:r[field] for o,field in OBJECTIVES.items()},
                tied_run_ids=json.loads(z['tied_run_ids_json'])))
        f.suptitle('같은 상황에서 서로 다른 목적의 선택')
        save(f,'09_objective_tradeoff',f'{sid}. 두 목적의 동점 집합이 겹치지 않고 경로 형태도 다른 실제 사례를 고정 해시로 선택했다. 42개 후보와 계산 오차 범위에 한정된 관찰이며 모든 상황의 차이를 대표하지 않는다.')
    write_json(OUT/'representative_equations.json',records);write_json(OUT/'tradeoff_equations.json',trade_json)
    table('case_integral_replay',checks);table('case_adaptive_nodes',node_rows)
    table('case_selection',[dict(run_id=r,role=t,reason=reason) for r,t,reason in selection])
    return len(trades)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    before=preserved_hashes()
    manifest_hash=sha(DATA/'dataset_manifest.json')
    m,s,name_mismatches=load()
    w,reps,checked=summarize(m,s)
    save,caps=figures(m,s,w,reps)
    trades=cases(m,w,save);table('figure_captions',caps)
    assert before==preserved_hashes(), 'Protected files changed'
    assert manifest_hash==sha(DATA/'dataset_manifest.json')
    write_json(OUT/'analysis_checks.json',dict(passed=True,scenario_count=len(s),candidate_count=len(m),
        main_scenarios=len(reps),excluded_unresolved=int(s.cohort.eq('excluded_unresolved').sum()),
        excluded_no_valid=int(s.cohort.eq('excluded_no_valid').sum()),winner_labels_replayed=checked,
        tradeoff_eligible_scenarios=trades,source_curve_label_mismatches=name_mismatches,
        preserved_files=len(before),protected_hashes=before,heldout_outcomes_read=False))
    write_json(OUT/'provenance.json',dict(dataset_version='phase4-mathematics-v3',dataset_manifest_sha256=manifest_hash,
        generation_signature=json.loads((DATA/'generation_state.json').read_text())['signature'],
        analysis_protocol_sha256=sha(ROOT/'docs/phase4_v3_analysis_protocol.md'),analysis_seed=SEED,
        script_sha256=sha(Path(__file__)),python=sys.version,executable=sys.executable,platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,source='train completed receipts only',
        curve_labels='Canonical curve_key mapping for display; original curve_name_ko preserved in raw data',
        numerical_verification='../phase4_v3_release_quality/verification.json'))
    print('Analysis complete',json.dumps({k:v for k,v in json.loads((OUT/'analysis_checks.json').read_text(encoding='utf-8')).items() if k!='protected_hashes'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    if '--refresh-overview-figures' in sys.argv or '--refresh-case-figures' in sys.argv:
        # Layout-only refresh from the already selected train analysis tables.
        provenance=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
        assert provenance['dataset_manifest_sha256']==sha(DATA/'dataset_manifest.json')
        m=pd.read_parquet(OUT/'train_analysis_metrics.parquet')
        assert set(m.data_split)=={'train'}
        read=lambda name:pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip')
        if '--refresh-overview-figures' in sys.argv:
            figures(m,read('scenario_exclusions'),read('main_winners'),read('relationship_representatives'))
        else:
            plt=plot_setup()
            def save(fig,name,caption):
                fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
                fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
                print('Refreshed',name,flush=True)
            cases(m,read('main_winners'),save)
        provenance['visual_refresh_script_sha256']=sha(Path(__file__))
        provenance['visual_refresh_note']='Figure layout/witness annotation refresh; selection, calculations and cohorts unchanged'
        write_json(OUT/'provenance.json',provenance)
    else:main()
