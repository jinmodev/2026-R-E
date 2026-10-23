"""Complete winner dossier from the original exported case, including ties.

Stages: --compute (project scientific runtime), --finalize (stdlib).
PDF is produced by render_case1_winner_bundle.py using the bundled runtime.
"""
from __future__ import annotations
import argparse,copy,csv,hashlib,html,json,math,re,shutil,sys,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'outputs/case1_winner_bundle_20261023'
OBJECTIVES={
 'distance':('경로 길이','path_length_units','unit'),
 'joint_effort':('관절 노력도','joint_velocity_effort','rad²/s'),
 'acceleration':('각가속도 비용','joint_acceleration_effort','rad²/s³'),
 'jerk':('관절 저크 비용','joint_jerk_effort','rad²/s⁵'),
 'curvature':('정칙화 곡률 비용','regularized_curvature_energy_inv_unit','1/unit'),
 'obstacle_penalty':('장애물 접근 벌점','obstacle_penalty_s_per_unit2','s/unit²'),
 'model_energy':('모형 에너지','model_electrical_energy_j','J'),
 'rms_torque':('RMS 토크','rms_joint_torque_nm','N·m'),
 'report_balanced':('종합 비용','report_balanced_cost_0_100','점'),
}
FAMILIES={'line':'직선','arc':'원호','bezier':'베지어','hermite':'에르미트','cycloid':'사이클로이드','limacon':'리마송'}
COLORS={'arc':'#3563e9','bezier':'#df9222','hermite':'#8155b9','cycloid':'#d94d88','limacon':'#079c92'}

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_csv(p):
    def typed(v):
        if v is None or v=='':return None
        if v in ('true','false'):return v=='true'
        try:return float(v)
        except ValueError:return v
    with p.open(encoding='utf-8-sig',newline='') as f:return [{k:typed(v) for k,v in r.items()} for r in csv.DictReader(f)]
def write_csv(p,rows):
    assert rows,p
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:('true' if v else 'false') if isinstance(v,bool) else v for k,v in r.items()})
def near(a,b):return math.isclose(float(a),float(b),rel_tol=2e-10,abs_tol=1e-9)
def short(rid):return rid.replace('interactive_','')
def plain(text):
    text=re.sub(r'<sub>(.*?)</sub>',r'_\1',text)
    text=re.sub(r'<super>(.*?)</super>',r'^(\1)',text)
    return html.unescape(re.sub('<[^>]+>','',text))
def poly_string(coeff,var='u'):
    terms=[]
    for power,v in enumerate(coeff):
        if abs(v)<1e-12:continue
        terms.append(f'{v:+.12g}'+('' if power==0 else var if power==1 else f'{var}<super>{power}</super>'))
    return ''.join(terms).lstrip('+') or '0'

def compute(out,refresh=False):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.integrate import quad
    sys.path.insert(0,str(ROOT/'src'))
    from robot_planner.curves import make_curve
    from robot_planner.kinematics import PlanarArm2R
    from robot_planner.evaluation import CircleObstacle
    from robot_planner.report_evaluation import compare_candidates,score_report_candidates,REPORT_WEIGHTS
    from robot_planner.report_math import polynomial_coefficients,cycloid_length
    from robot_planner.report_physics import ReportPhysicsConfig,curve_arrays,joint_arrays,torque_arrays
    from robot_planner.phase4_v3 import adaptive_joint_costs
    assert not out.exists() or (refresh and (out/'dossier.json').is_file()),'Use a new output path, or --refresh for this generated dossier.'
    source=ROOT/'분석자료_1';folders=list(source.glob('robot_path_comparison_*'));assert len(folders)==1
    folder=folders[0];source_files=[p for p in source.rglob('*') if p.is_file()]
    hashes={str(p.relative_to(source)):digest(p) for p in source_files}
    original=read_csv(folder/'candidate_metrics.csv');labels=read_csv(folder/'objective_winners.csv')
    equations={r['run_id']:r for r in read_csv(folder/'equations.csv')}
    cfg_json=json.loads((source/'robot_path_run.json').read_text(encoding='utf-8'))
    inp=cfg_json['inputs'];cfg=ReportPhysicsConfig(**cfg_json['physics_model'])
    arm=PlanarArm2R(inp['link1'],inp['link2'],(inp['base_x'],inp['base_y']))
    obstacles=tuple(CircleObstacle(**o) for o in cfg_json['obstacles'])
    start=(inp['start_x'],inp['start_y']);goal=(inp['goal_x'],inp['goal_y'])
    fresh=compare_candidates(arm,start,goal,obstacles=obstacles,duration_s=3,physics_config=cfg)
    fresh_rows={r['run_id']:r for r in fresh['records']};byid={r['run_id']:r for r in original}
    for row in original:
        now=fresh_rows[row['run_id']]
        assert row['candidate_status']==now['candidate_status'] and row['selection_valid']==now['selection_valid']
        for _,field,_ in OBJECTIVES.values():
            assert (row[field] is None and now[field] is None) or near(row[field],now[field]),(row['run_id'],field)
    for old,new in zip(labels,fresh['labels']):
        for k in ('objective','label_status','winner_run_id','tie_count','tied_run_ids_json'):assert old[k]==new[k],k
    assert len(original)==42 and len(labels)==9 and all(r['label_status']=='ready' for r in labels)
    selected=[]
    for r in labels:
        for rid in [r['winner_run_id']]+json.loads(r['tied_run_ids_json']):
            if rid not in selected:selected.append(rid)
    assert len(selected)==6
    valid=[r for r in original if r['selection_valid']]
    eligible=[r for r in valid if r['energy_selection_ready']]
    assert len(valid)==len(eligible)==12
    out.mkdir(parents=True,exist_ok=True)
    for d in ('figures','data','integration_steps','provenance_inputs','code_snapshot'):(out/d).mkdir(exist_ok=True)
    for p in source_files:
        dest=out/'provenance_inputs'/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in (ROOT/'src/robot_planner').glob('*.py'):shutil.copy2(p,out/'code_snapshot'/p.name)
    geometry={rid:[] for rid in selected};time_rows={rid:[] for rid in selected}
    for filename,target in [('geometry_samples.csv',geometry),('time_samples.csv',time_rows)]:
        for row in read_csv(folder/filename):
            if row['run_id'] in target:target[row['run_id']].append(row)
    for target in (geometry,time_rows):
        for rows in target.values():
            rows.sort(key=lambda r:r['sample_index']);assert len(rows)==161
            assert [r['sample_index'] for r in rows]==list(range(161))
    trap_rows=[];node_rows=[];score_rows=[];reference_rows=[];summary=[];items=[];sample_points=[]
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(out/'figures'/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
        fig.savefig(out/'figures'/f'{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)
    def trap(rid,field,x,y,grid):
        total=0.;parts=[]
        for i,(a,b,f0,f1) in enumerate(zip(x,x[1:],y,y[1:])):
            v=float((b-a)*(f0+f1)/2);parts.append(v);total+=v
            trap_rows.append(dict(run_id=rid,metric=field,grid=grid,interval=i,left=float(a),right=float(b),
                                  integrand_left=float(f0),integrand_right=float(f1),contribution=v,cumulative=total))
        return math.fsum(parts)
    for index,rid in enumerate(selected,1):
        r=byid[rid];result=fresh['results'][rid];metrics=result.metrics
        curve=make_curve(r['curve_key'],start,goal,shape=r['shape'],samples=161)
        parameter=dict(curve.parameters);coeff=polynomial_coefficients(curve)
        mode='위' if r['elbow_mode']=='up' else '아래'
        display=f"{FAMILIES[r['curve_key']]} {r['shape']:+.1f} {mode}"
        reasons=[OBJECTIVES[l['objective']][0]+(' 공동 1등' if l['tie_count']>1 else ' 1등') for l in labels if rid in json.loads(l['tied_run_ids_json'])]
        equations_display=[];derivatives=[];domain=['0 ≤ u ≤ 1','0 ≤ t ≤ 3 s']
        derivative_variable='u'
        if coeff:
            equations_display=[f'{axis}(u)={poly_string(c)}' for axis,c in zip(('x','y'),coeff)]
            for axis,c in zip(('x','y'),coeff):
                derivatives.extend([f'{axis}′(u)={poly_string([j*c[j] for j in range(1,4)])}',
                                    f'{axis}″(u)={poly_string([j*(j-1)*c[j] for j in range(2,4)])}'])
            length_recipe='L=∫₀¹ √[x′(u)²+y′(u)²] du'
            if r['curve_key']=='bezier':
                aa=sum((2*c[2])**2 for c in coeff);bb=2*sum(c[1]*2*c[2] for c in coeff);cc=sum(c[1]**2 for c in coeff)
                z0=bb/(2*aa);k2=(4*aa*cc-bb*bb)/(4*aa*aa)
                def primitive(u):
                    z=u+z0;return math.sqrt(aa)/2*(z*math.sqrt(z*z+k2)+k2*math.asinh(z/math.sqrt(k2)))
                length_ref=primitive(1)-primitive(0)
                length_recipe=f'L=∫₀¹√({aa:.12g}u²+({bb:.12g})u+{cc:.12g})du; 平方完成'.replace('平方完成','완전제곱식으로 정리한 뒤 로그 부정적분')
            else:
                length_ref=quad(lambda u:float(np.linalg.norm(curve_arrays(curve,np.array([u]))[1][0])),0,1,epsabs=1e-11,epsrel=1e-12)[0]
        elif r['curve_key']=='arc':
            p=parameter;a0=p['theta_start'];a1=a0+p['theta_delta']
            domain += [f'α: {a0:.12g} → {a1:.12g} rad (감소 방향)']
            equations_display=[f'α(u)={a0:.12g}+({p["theta_delta"]:.12g})u',
                               f'x(u)={p["center_x"]:.12g}+{p["radius"]:.12g}cos α(u)',
                               f'y(u)={p["center_y"]:.12g}+{p["radius"]:.12g}sin α(u)']
            derivatives=[f'x′=-RΔα sin α,　y′=RΔα cos α',
                         'x″=-R(Δα)²cos α,　y″=-R(Δα)²sin α']
            length_ref=p['radius']*abs(p['theta_delta']);length_recipe=f'L=R|Δα|={p["radius"]:.12g}×{abs(p["theta_delta"]):.12g}'
        elif r['curve_key']=='limacon':
            p=parameter;phi0=p['phase'];beta=math.atan2(math.sin(p['rotation']),math.cos(p['rotation']));scale=p['scale']
            rho=p['a']+p['b']*math.cos(phi0);x0=rho*math.cos(phi0);y0=p['side']*rho*math.sin(phi0)
            cx=start[0]-scale*(math.cos(beta)*x0-math.sin(beta)*y0);cy=start[1]-scale*(math.sin(beta)*x0+math.cos(beta)*y0)
            assert p['side']==1
            equations_display=['φ(u)=-3π/5+(6π/5)u','r(φ)=1+0.79cos φ',
                               f'x(φ)={cx:.12g}+{scale:.12g}r(φ)cos(φ+{beta:.12g})',
                               f'y(φ)={cy:.12g}+{scale:.12g}r(φ)sin(φ+{beta:.12g})']
            domain+=['-3π/5 ≤ φ ≤ 3π/5 (증가 방향)'];derivative_variable='φ'
            derivatives=['r′=-0.79sin φ,　r″=-0.79cos φ',
                         'x′=s[r′cos(φ+β)-r sin(φ+β)]',
                         'y′=s[r′sin(φ+β)+r cos(φ+β)]']
            length_ref=scale*quad(lambda ph:math.sqrt(1.6241+1.58*math.cos(ph)),-3*math.pi/5,3*math.pi/5,epsabs=1e-11)[0]
            length_recipe=f'L={scale:.12g}∫[-3π/5,3π/5]√(1.6241+1.58cos φ)dφ (정밀 수치적분)'
        else:
            assert r['curve_key']=='cycloid'
            D=math.dist(start,goal);A=D/(2*math.pi);B=parameter['height']/2;ex=(goal[0]-start[0])/D;ey=(goal[1]-start[1])/D
            equations_display=['φ=2πu',f'X(φ)={A:.12g}(φ-sin φ)',f'Y(φ)={B:.12g}(1-cos φ)',
                               f'x(φ)=2+({ex:.12g})X+({-ey:.12g})Y',f'y(φ)=({ey:.12g})X+({ex:.12g})Y']
            domain+=['0 ≤ φ ≤ 2π (증가 방향)'];derivative_variable='φ'
            derivatives=['X′=A(1-cos φ),　Y′=B sin φ','X″=A sin φ,　Y″=B cos φ']
            length_ref=cycloid_length(A,B);length_recipe='c=√(B²-A²),　L=4[B+(A²/c)ln((B+c)/A)]'
        g=geometry[rid];ts=time_rows[rid]
        L=trap(rid,'L',[x['geometry_u'] for x in g],[x['path_speed_units_per_u'] for x in g],'cosine_u_161')
        K=trap(rid,'Kepsilon',[x['geometry_u'] for x in g],[x['regularized_curvature_integrand_inv_unit'] for x in g],'cosine_u_161')
        O=trap(rid,'O',[x['time_s'] for x in ts],[x['obstacle_penalty_integrand_per_unit2'] for x in ts],'uniform_t_161')
        assert near(L,r['path_length_units']) and near(K,r['regularized_curvature_energy_inv_unit']) and near(O,r['obstacle_penalty_s_per_unit2'])
        adaptive=adaptive_joint_costs(arm,curve,3,r['elbow_mode']);assert adaptive['status']=='accepted'
        adaptive_data={}
        for metric in ('joint_velocity_effort','joint_acceleration_effort','joint_jerk_effort'):
            panels=[p for p in adaptive['panels'] if p['metric']==metric]
            contributions=[];nodes=[]
            for panel_index,panel in enumerate(panels):
                for node_index,(t,w,f) in enumerate(zip(panel['time_s'],panel['weights_s'],panel['integrand'])):
                    v=w*f;contributions.append(v);nodes.append((t,f))
                    node_rows.append(dict(run_id=rid,metric=metric,panel=panel_index,node=node_index,
                                          panel_left=panel['lo'],panel_right=panel['hi'],time_s=t,weight_s=w,integrand=f,contribution=v))
            total=math.fsum(contributions);assert near(total,r[metric]),(rid,metric,total,r[metric])
            adaptive_data[metric]=dict(panels=len(panels),nodes=len(nodes),sum=total,quad_error_estimate=adaptive['diagnostics'][metric]['quad_error_estimate'])
        state=joint_arrays(arm,curve,np.linspace(0,3,641),3,r['elbow_mode'],cfg)
        physical=torque_arrays(arm,state['theta'],state['velocity'],state['acceleration'],cfg)
        physics_integrals={}
        for key in ('net_power_w','positive_power_w','absorbed_power_w','copper_loss_power_w','torque_squared_nm2'):
            physics_integrals[key]=trap(rid,key,state['time_s'],physical[key],'uniform_t_641')
        assert near(physics_integrals['positive_power_w']+physics_integrals['copper_loss_power_w'],r['model_electrical_energy_j'])
        assert near(math.sqrt(physics_integrals['torque_squared_nm2']/3),r['rms_joint_torque_nm'])
        eps=metrics['curvature_speed_epsilon_units_per_u']
        def kfun(u):
            _,d,dd=curve_arrays(curve,np.array([u]));v=d[0];a=dd[0]
            return float((v[0]*a[1]-v[1]*a[0])**2/(float(v@v)+eps*eps)**2.5)
        Kref=quad(kfun,0,1,points=[.00001,.0001,.001,.01,.1,.5,.9,.99,.999,.9999,.99999],epsabs=1e-10,epsrel=1e-10,limit=300)[0]
        reference_rows.extend([dict(run_id=rid,metric=field,source_value=val,reference_value=ref,
                                    absolute_difference=abs(val-ref),relative_difference=abs(val-ref)/abs(ref) if ref else None,
                                    reference_method=method) for field,val,ref,method in [
                                        ('path_length_units',r['path_length_units'],length_ref,'closed form or independent adaptive curve integral'),
                                        ('regularized_curvature_energy_inv_unit',r['regularized_curvature_energy_inv_unit'],Kref,'independent adaptive integral, same u and epsilon')]])
        scores=[]
        for field,w in REPORT_WEIGHTS.items():
            lo=min(x[field] for x in eligible);hi=max(x[field] for x in eligible)
            n=(r[field]-lo)/(hi-lo) if hi>lo else 0.;deduct=100*w*n
            assert near(n,r['report_normalized_'+field])
            entry=dict(run_id=rid,metric=field,raw_value=r[field],minimum=lo,maximum=hi,
                       normalized=n,weight=w,deduction_points=deduct,denominator_candidates=len(eligible))
            scores.append(entry);score_rows.append(entry)
        cost=math.fsum(x['deduction_points'] for x in scores);score=100-cost
        assert near(cost,r['report_balanced_cost_0_100']) and near(score,r['report_balanced_reward_0_100'])
        u_dense=np.linspace(0,1,1001);xy=curve_arrays(curve,u_dense)[0]
        assert np.max(np.abs(xy[0]-start))<1e-10 and np.max(np.abs(xy[-1]-goal))<1e-10
        p5=curve_arrays(curve,np.array([0,.25,.5,.75,1]))[0]
        for u,pt in zip([0,.25,.5,.75,1],p5):sample_points.append(dict(run_id=rid,u=u,x=float(pt[0]),y=float(pt[1])))
        error=max(x['max_forward_kinematics_error_units'] for x in [r])
        endpoints={name:dict(time_s=ts[ix]['time_s'],theta1_rad=ts[ix]['theta1_rad'],theta2_rad=ts[ix]['theta2_rad']) for name,ix in [('start',0),('end',-1)]}
        item=dict(index=index,run_id=rid,short_id=short(rid),display=display,reasons=reasons,curve_key=r['curve_key'],shape=r['shape'],elbow_mode=r['elbow_mode'],
                  equations_display=equations_display,derivatives=derivatives,derivative_variable=derivative_variable,domains=domain,
                  polynomial_coefficients=coeff,parameters=parameter,length_recipe=length_recipe,length_reference=length_ref,Kepsilon_reference=Kref,
                  metrics={k:r[k] for _,k,_ in OBJECTIVES.values()},score=score,scores=scores,adaptive=adaptive_data,physics_integrals=physics_integrals,
                  positive_work=r['positive_mechanical_work_j'],copper=r['motor_copper_loss_j'],net_work=r['net_mechanical_work_j'],
                  end_angles=endpoints,epsilon_u=eps,forward_error=error,
                  figure=f'figures/{index:02d}_{short(rid)}_path.png',integrands_figure=f'figures/{index:02d}_{short(rid)}_integrands.png')
        items.append(item)
        summary.append(dict(run_id=rid,name_ko=display,winner_of='; '.join(reasons),start_x=start[0],start_y=start[1],goal_x=goal[0],goal_y=goal[1],
                            u_min=0,u_max=1,t_min_s=0,t_max_s=3,shape=r['shape'],elbow_mode=r['elbow_mode'],
                            **{k:r[k] for _,k,_ in OBJECTIVES.values()},report_balanced_reward_0_100=score,
                            min_obstacle_clearance_units=r['min_obstacle_clearance_units']))
        color=COLORS[r['curve_key']];style='--' if r['curve_key']=='bezier' and r['shape']<0 else '-'
        fig,ax=plt.subplots(figsize=(7.6,4.3));ax.plot(xy[:,0],xy[:,1],color=color,linestyle=style,lw=2.4,label=display)
        for at in (.2,.5,.8):
            k=int(at*1000);ax.annotate('',xy=xy[k+12],xytext=xy[k-12],arrowprops=dict(arrowstyle='->',color=color,lw=1.6))
        for o in obstacles:
            ax.add_patch(plt.Circle((o.center_x,o.center_y),o.radius_units,fc='#fbdede',ec='#d64545',lw=1.3))
            ax.text(o.center_x,o.center_y,o.obstacle_id,ha='center',va='center',fontsize=9)
        ax.scatter(*start,s=55,marker='s',c='#172033',zorder=5);ax.annotate('시작 u=0',(start),xytext=(8,6),textcoords='offset points',fontsize=9)
        ax.scatter(*goal,s=100,marker='*',c='#f59e0b',zorder=5);ax.annotate('도착 u=1',(goal),xytext=(-50,-17),textcoords='offset points',fontsize=9)
        ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='x [unit]',ylabel='y [unit]',title=f'{short(rid)} · {display}')
        ax.grid(alpha=.17);ax.margins(.17);fig.tight_layout();save(fig,f'{index:02d}_{short(rid)}_path')
        fig,axes=plt.subplots(4,2,figsize=(9,11));fig.suptitle(f'{short(rid)} · 평가 피적분함수',fontsize=13)
        plots=[('L: 경로 속도','u',[x['geometry_u'] for x in g],[x['path_speed_units_per_u'] for x in g],'unit/u')]
        for metric,title,unit in [('joint_velocity_effort','Eθ: 관절 각속도 제곱합','rad^2/s^2'),('joint_acceleration_effort','A: 관절 각가속도 제곱합','rad^2/s^4'),('joint_jerk_effort','J: 관절 각저크 제곱합','rad^2/s^6')]:
            nodes=sorted((x['time_s'],x['integrand']) for x in node_rows if x['run_id']==rid and x['metric']==metric)
            plots.append((title,'t [s]',[x[0] for x in nodes],[x[1] for x in nodes],unit))
        plots.extend([('Kε: 정칙화 곡률 피적분함수','u',[x['geometry_u'] for x in g],[x['regularized_curvature_integrand_inv_unit'] for x in g],'1/unit'),
                      ('O: 장애물 접근 벌점률','t [s]',[x['time_s'] for x in ts],[x['obstacle_penalty_integrand_per_unit2'] for x in ts],'1/unit²'),
                      ('E: 모형 에너지 비용률','t [s]',state['time_s'],physical['positive_power_w']+physical['copper_loss_power_w'],'W'),
                      ('RMS: 관절 토크 제곱합','t [s]',state['time_s'],physical['torque_squared_nm2'],'(N·m)²')])
        for ax,(title,xlabel,x,y,ylabel) in zip(axes.flat,plots):
            ax.plot(x,y,color=color,lw=1.2);ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.17)
            # Do not magnify roundoff in analytically constant integrands.
            values=np.asarray(y,dtype=float)
            center=float(np.mean(values))
            if float(np.ptp(values)) < 1e-9*max(1.,abs(center)):
                padding=max(.05,abs(center)*.1)
                ax.set_ylim(center-padding,center+padding)
                ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        fig.tight_layout(rect=(0,0,1,.97));save(fig,f'{index:02d}_{short(rid)}_integrands')
        physics_rows=[dict(run_id=rid,sample_index=k,time_s=float(t),theta1_rad=float(state['theta'][k,0]),theta2_rad=float(state['theta'][k,1]),
                           **{field:float(values[k]) for field,values in physical.items()}) for k,t in enumerate(state['time_s'])]
        write_csv(out/'data'/f'{short(rid)}_physics641.csv',physics_rows)
        print(f'Completed {index}/{len(selected)}: {short(rid)}',flush=True)
    fig,ax=plt.subplots(figsize=(9,7))
    for item in items:
        c=make_curve(item['curve_key'],start,goal,shape=item['shape']);xy=curve_arrays(c,np.linspace(0,1,1001))[0]
        ax.plot(xy[:,0],xy[:,1],lw=2,color=COLORS[item['curve_key']],linestyle='--' if item['curve_key']=='bezier' and item['shape']<0 else '-',label=item['short_id'])
    for o in obstacles:ax.add_patch(plt.Circle((o.center_x,o.center_y),o.radius_units,fc='#fbdede',ec='#d64545'))
    ax.scatter(*start,marker='s',s=55,c='#172033',zorder=4);ax.scatter(*goal,marker='*',s=120,c='#f59e0b',zorder=4)
    ax.annotate('시작',(start),xytext=(8,4),textcoords='offset points');ax.annotate('도착',(goal),xytext=(-30,-18),textcoords='offset points')
    ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='x [unit]',ylabel='y [unit]',title='기준별 우승 후보 6개 · 공동 1등 포함')
    ax.legend(fontsize=9,loc='upper left',bbox_to_anchor=(1.01,1));ax.grid(alpha=.17);fig.tight_layout();save(fig,'00_all_winners')
    write_csv(out/'우승후보_전체평가값.csv',summary);write_csv(out/'기준별_우승라벨.csv',labels)
    write_csv(out/'종합점수_계산과정.csv',score_rows);write_csv(out/'정밀검산_비교.csv',reference_rows)
    write_csv(out/'data/곡선_진행률별_좌표.csv',sample_points)
    write_csv(out/'data/우승후보_정밀원본식.csv',[equations[rid] for rid in selected])
    write_csv(out/'data/우승후보_시간표본161.csv',[x for rid in selected for x in time_rows[rid]])
    write_csv(out/'data/우승후보_기하표본161.csv',[x for rid in selected for x in geometry[rid]])
    write_csv(out/'integration_steps/사다리꼴_구간별_계산.csv',trap_rows)
    write_csv(out/'integration_steps/관절적응형_가중적분_계산.csv',node_rows)
    payload=dict(input=inp,obstacles=cfg_json['obstacles'],physics=cfg_json['physics_model'],labels=labels,items=items,
                 candidate_count=42,valid_candidates=12,normalization_candidates=12,source_hashes=hashes,
                 reference_rows=reference_rows,weights=REPORT_WEIGHTS,
                 verification=dict(all_42_metrics_replayed=True,all_9_labels_replayed=True,all_ties_included=True,
                    selected_count=6,geometric_and_obstacle_sums_replayed=True,adaptive_node_sums_replayed=True,
                    physics641_sums_replayed=True,normalization_and_scores_replayed=True,
                    source_files_unchanged=hashes=={str(p.relative_to(source)):digest(p) for p in source_files},
                    trap_interval_rows=len(trap_rows),adaptive_node_rows=len(node_rows),
                    maximum_stored_forward_error=max(item['forward_error'] for item in items)))
    assert payload['verification']['source_files_unchanged']
    (out/'dossier.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    (out/'먼저읽기.txt').write_text('기존 분석자료_1을 기준으로 9개 평가기준의 우승 후보와 공동 우승 후보를 정리했습니다.\n'
       '우승곡선_전체자료.pdf를 먼저 보세요. 우승곡선_전체자료.txt는 복사용입니다.\n'
       'figures: 후보 6개의 개별 경로 및 8개 피적분함수 그래프, 전체 비교 그래프. PNG 300dpi와 SVG입니다.\n'
       'integration_steps: 실제 적분의 모든 구간 또는 가중 노드와 기여값을 CSV로 제공합니다.\n'
       '곡선 구간은 모든 후보 u=0부터1, 시간 t=0부터3초이며 시작(2,0), 도착(-4.8,-1.5)입니다.\n'
       '각 우승은 준비된 42개 안에서의 최적입니다. 곡률 공동 우승 후보도 포함했습니다.\n'
       '개별 평가값은 작은 값이 우수한 비용이며, 종합점수만 클수록 좋습니다.\n'
       '정밀 길이/곡률 검산값은 기존 데이터와 구분했습니다. 기존 값·라벨·가중치는 변경하지 않았습니다.\n'
       '원자료와 소스 파일은 보존되며 provenance_inputs와 code_snapshot에서 확인할 수 있습니다.\n'
       '에너지는 가정한 2R·모터 모형 값이며 실측이 아닙니다. 에너지 결과를 실제 장치에 일반화하지 마세요.\n'
       '재실행: .venv/Scripts/python.exe scripts/build_case1_winner_bundle.py --compute --output 새출력폴더\n'
       'PDF: 제공된 Python으로 scripts/render_case1_winner_bundle.py --output 같은출력폴더\n'
       '정리: .venv/Scripts/python.exe scripts/build_case1_winner_bundle.py --finalize --output 같은출력폴더\n',encoding='utf-8-sig')
    print(json.dumps(payload['verification'],ensure_ascii=False),flush=True)

def finalize(out):
    data=json.loads((out/'dossier.json').read_text(encoding='utf-8'))
    checks=json.loads((out/'pdf_checks.json').read_text(encoding='utf-8'))
    assert checks['visual_review_pending'] is False, 'Complete visual review before packaging.'
    assert json.loads((out/'independent_flatfile_audit.json').read_text(encoding='utf-8'))['status']=='passed'
    for name,expected in data['source_hashes'].items():
        assert digest(ROOT/'분석자료_1'/name)==expected
        assert digest(out/'provenance_inputs'/name)==expected
    for name in ('build_case1_winner_bundle.py','render_case1_winner_bundle.py','audit_case1_winner_bundle.py'):shutil.copy2(ROOT/'scripts'/name,out/'code_snapshot'/name)
    for filename in ('우승곡선_전체자료.pdf','우승곡선_전체자료.txt'):assert (out/filename).is_file()
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json']
    manifest={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[out/'manifest.json']:z.write(p,str(p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,source_preservation=True),ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--compute',action='store_true');ap.add_argument('--refresh',action='store_true');ap.add_argument('--finalize',action='store_true')
    args=ap.parse_args()
    if args.compute:compute(args.output,args.refresh)
    if args.finalize:finalize(args.output)
