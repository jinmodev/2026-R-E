"""Two-candidate mathematical derivation and full metric replay, then PDF."""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/analysis_case1_20261021'
TARGETS=['interactive_limacon_p80_down','interactive_cycloid_p80_up']

def compute():
    from scipy.integrate import quad
    sys.path.insert(0,str(ROOT/'src'))
    from robot_planner.curves import make_curve
    from robot_planner.kinematics import PlanarArm2R
    from robot_planner.evaluation import CircleObstacle
    from robot_planner.report_evaluation import simulate_report_trajectory
    from robot_planner.report_physics import ReportPhysicsConfig
    src=ROOT/'분석자료_1';j=json.loads((src/'robot_path_run.json').read_text(encoding='utf-8'))
    f=next(src.glob('robot_path_comparison_*/candidate_metrics.csv'))
    with f.open(encoding='utf-8-sig',newline='') as stream:rows={r['run_id']:r for r in csv.DictReader(stream)}
    results={}
    check_fields=['path_length_units','joint_velocity_effort','joint_acceleration_effort','joint_jerk_effort',
                  'regularized_curvature_energy_inv_unit','obstacle_penalty_s_per_unit2','min_obstacle_clearance_units',
                  'model_electrical_energy_j','rms_joint_torque_nm','net_mechanical_work_j']
    for rid in TARGETS:
        r=rows[rid];arm=PlanarArm2R(float(r['link1']),float(r['link2']),(float(r['base_x']),float(r['base_y'])))
        curve=make_curve(r['curve_key'],(float(r['start_x']),float(r['start_y'])),(float(r['goal_x']),float(r['goal_y'])),shape=float(r['shape']))
        result=simulate_report_trajectory(arm,curve,r['elbow_mode'],obstacles=tuple(CircleObstacle(**o) for o in j['obstacles']),duration_s=3,physics_config=ReportPhysicsConfig(**j['physics_model']))
        for field in check_fields:assert math.isclose(float(r[field]),result.metrics[field],rel_tol=2e-10,abs_tol=1e-9),(rid,field)
        m=result.metrics
        results[r['curve_key']]={k:m[k] for k in check_fields+['positive_mechanical_work_j','absorbed_mechanical_work_j','motor_copper_loss_j','damping_loss_j','potential_energy_change_j','work_energy_residual_j','physics_grids_json']}
        results[r['curve_key']]['score']=float(r['report_balanced_reward_0_100'])
    scale=4.843272276019263;b=.79;alpha=3*math.pi/5;omega=6*math.pi/5
    eps=.001*math.sqrt(48.49);A=math.sqrt(48.49)/(2*math.pi);B=.2*math.sqrt(48.49);c=math.sqrt(B*B-A*A)
    S=lambda f:1+b*b+2*b*math.cos(f)
    N=lambda f:1+2*b*b+3*b*math.cos(f)
    references={
        'length_limacon':scale*quad(lambda f:math.sqrt(S(f)),-alpha,alpha,epsabs=1e-11,epsrel=1e-13)[0],
        'length_cycloid':4*(B+A*A/c*math.log((B+c)/A)),
        'K0_limacon':quad(lambda f:N(f)**2/S(f)**2.5,-alpha,alpha,epsabs=1e-12)[0]/scale,
        'Keps_limacon':quad(lambda f:N(f)**2/(S(f)+(eps/(scale*omega))**2)**2.5,-alpha,alpha,epsabs=1e-12)[0]/scale,
        'epsilon_u':eps,'A':A,'B':B,'c':c,
    }
    def integrand(f):
        return A*A*B*B*(1-math.cos(f))**2/(A*A*(1-math.cos(f))**2+B*B*math.sin(f)**2+(eps/(2*math.pi))**2)**2.5
    references['Keps_cycloid']=2*quad(integrand,0,math.pi,epsabs=1e-10,epsrel=1e-10,points=[.0001,.001,.005,.01,.1])[0]
    # Ordinary squared curvature diverges logarithmically at both cycloid cusps.
    references['K0_cycloid']='diverges at cusp endpoints'
    payload={'results':results,'references':references,'input':j['inputs'],'physics':j['physics_model'],
             'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'replay':'passed',
             'reference_method':'scipy.integrate.quad, unchanged parameters; logarithmic closed length for cycloid',
             'source_values_unchanged':True}
    (OUT/'전체평가_검산.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(references,ensure_ascii=False))

def pdf():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,KeepTogether
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from pypdf import PdfReader
    data=json.loads((OUT/'전체평가_검산.json').read_text(encoding='utf-8'))
    m=data['results'];ref=data['references']
    pdfmetrics.registerFont(TTFont('Malgun','C:/Windows/Fonts/malgun.ttf'))
    pdfmetrics.registerFont(TTFont('MalgunBold','C:/Windows/Fonts/malgunbd.ttf'))
    body=ParagraphStyle('body',fontName='Malgun',fontSize=10.5,leading=16,spaceAfter=7)
    eqstyle=ParagraphStyle('eq',parent=body,fontSize=10.5,leading=18,leftIndent=8,spaceAfter=4)
    head=ParagraphStyle('h',fontName='MalgunBold',fontSize=12,leading=18,spaceBefore=12,spaceAfter=7)
    title=ParagraphStyle('title',fontName='MalgunBold',fontSize=17,leading=23,spaceAfter=12)
    note=ParagraphStyle('note',parent=body,fontSize=9,leading=14,textColor=HexColor('#475569'))
    story=[]
    def normalize(t):
        return (t.replace('−','-').replace('θᵢ','θ<sub>i</sub>').replace('Θᵢ','Θ<sub>i</sub>')
                .replace('θ̇','θ′').replace('θ̈','θ″').replace('‴',"'''")
                .replace('≈','~').replace('⁵','<super>5</super>').replace('₀','<sub>0</sub>'))
    def p(t,s=body):
        story.append(Paragraph(normalize(t),s))
    def e(t):p(t,eqstyle)
    def h(t):p(t,head)
    def page(t):
        if story:story.append(PageBreak())
        p(t,title)
    def table(headers,rows,widths):
        cells=[[Paragraph(normalize(str(x)),note) for x in r] for r in [headers]+rows]
        tab=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
        tab.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.5,HexColor('#D9D9D9')),('BACKGROUND',(0,0),(-1,0),HexColor('#E8EDF5')),
                                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([tab,Spacer(1,8)])
    def values(fields):
        return [[name,f"{m['limacon'][field]:.8f}",f"{m['cycloid'][field]:.8f}"] for name,field in fields]

    page('두 곡선의 미분과 길이 계산')
    p('리마송 아래 해와 사이클로이드 위 해를 비교한다. 손으로 유도한 식과 수치계산 결과를 구분했다. 모든 평가는 동일한 입력의 준비된 42개 후보 안에서의 비교다.')
    p('밑부분 (0, 0), 링크 길이 5.48532 및 6.13245 unit, 시작점 (2, 0), 도착점 (-4.8, -1.5), T=3 s. 장애물 중심 (-1.3, -2), 반지름 0.45 unit.',note)
    h('리마송')
    e('r=1+0.79 cos φ,　r′=-0.79 sin φ,　r″=-0.79 cos φ')
    e('s=4.84327227602,　β=1.78790764153,　-3π/5 ≤ φ ≤ 3π/5')
    e('x=C<sub>x</sub>+sr cos(φ+β),　y=C<sub>y</sub>+sr sin(φ+β)')
    e('x′=s[r′cos(φ+β)-r sin(φ+β)]')
    e('y′=s[r′sin(φ+β)+r cos(φ+β)]')
    e('x′²+y′²=s²(r²+r′²)=s²(1.6241+1.58 cos φ)')
    e('L=s ∫<sub>-3π/5</sub><super>3π/5</super> √(1.6241+1.58 cos φ) dφ')
    p(f"정밀 수치적분: L ≈ {ref['length_limacon']:.8f} unit. 일반적인 초등함수 부정적분이 아니라 타원적분 형태다.",note)
    h('일반화된 사이클로이드')
    e('X=A(φ-sin φ),　Y=B(1-cos φ),　0 ≤ φ ≤ 2π')
    e('A=1.10827164932802,　B=1.39269522868429')
    e('X′=A(1-cos φ),　Y′=B sin φ')
    e('X″=A sin φ,　Y″=B cos φ')
    e('L=∫<sub>0</sub><super>2π</super> √[A²(1-cos φ)²+B²sin²φ] dφ')
    e('z=cos(φ/2) 로 치환하면　L=8∫<sub>0</sub><super>1</super> √[A²+(B²-A²)z²] dz')
    e('c=√(B²-A²),　L=4[B+(A²/c)ln((B+c)/A)]')
    p(f"부정적분 결과에 대입: L ≈ {ref['length_cycloid']:.8f} unit. 회전·평행이동은 길이를 바꾸지 않는다. A≠B이므로 표준 공식 8A는 쓰지 않는다.",note)
    p('앞의 ′는 φ 미분이다. 길이는 시간함수와 무관한 기하량이다. 기존 데이터 길이는 161개 기하 표본의 사다리꼴 근삿값이므로 이 검산값과 구분한다.',note)

    page('곡률과 정칙화 곡률 비용')
    h('리마송의 보통 곡률')
    e('κ(φ)=|r²+2r′²-rr″| / [s(r²+r′²)<super>3/2</super>]')
    e('κ(φ)=|2.2482+2.37 cos φ| / [s(1.6241+1.58 cos φ)<super>3/2</super>]')
    p('이는 점마다의 휘어짐이다. 단위는 1/unit이며, 저크나 곡률 적분값 자체가 아니다.')
    h('사이클로이드의 보통 곡률')
    e('κ(φ)=AB(1-cos φ) / [A²(1-cos φ)²+B²sin²φ]<super>3/2</super>')
    p('0&lt;φ&lt;2π에서 위 식을 사용한다. 두 끝점은 속도가 0인 뾰족점으로 보통 곡률이 정의되지 않는다.')
    e('φ→0일 때　κ ≈ A/(2B²φ),　ds ≈ Bφ dφ')
    e('따라서　κ²ds ≈ [A²/(4B³)] dφ/φ')
    p('∫dφ/φ는 끝점에서 발산한다. 따라서 이 사이클로이드의 보통 곡률 제곱 적분 K₀=∫κ²ds는 무한대다. 단순 곡률 적분 ∫κ ds와 혼동하면 안 된다.')
    h('프로그램이 실제 사용하는 비용')
    e('K<sub>ε</sub>=∫<sub>0</sub><super>1</super> (x<sub>u</sub>′y<sub>u</sub>″-y<sub>u</sub>′x<sub>u</sub>″)²')
    e('　　　　　　 / (x<sub>u</sub>′²+y<sub>u</sub>′²+ε²)<super>5/2</super> du')
    e(f"ε=0.001×시작·도착 거리={ref['epsilon_u']:.14f} unit/u")
    p('여기의 ′는 u 미분이다. φ로 치환할 때 ε도 함께 변환해야 한다. 리마송은 φ=-3π/5+(6π/5)u, 사이클로이드는 φ=2πu이므로 φ 식의 속도 정칙화값은 각각 ε/(6π/5), ε/(2π)이다. ε를 그대로 옮겨 쓰면 다른 비용이 된다.',note)
    table(['곡선','저장 Kε [1/unit]','독립 정밀 적분 Kε'],[
        ['리마송',f"{m['limacon']['regularized_curvature_energy_inv_unit']:.8f}",f"{ref['Keps_limacon']:.8f}"],
        ['사이클로이드',f"{m['cycloid']['regularized_curvature_energy_inv_unit']:.8f}",f"{ref['Keps_cycloid']:.8f}"]],[120,180,211])
    p(f"리마송의 보통 제곱 곡률 적분 K₀ ≈ {ref['K0_limacon']:.8f} 1/unit. 독립 정밀 적분과 저장 Kε는 표본 근사 때문에 조금 다르다. 원자료와 기존 순위는 바꾸지 않았다.",note)

    page('관절 노력도와 가속도와 저크')
    p('끝점 좌표만 미분한 값과 관절각을 미분한 값은 다르다. 역기구학으로 두 관절각을 구한 뒤 시간에 대해 미분한다.')
    h('관절각 합성식')
    e('c(t)=[x(t)²+y(t)²-L₁²-L₂²]/(2L₁L₂)')
    e('θ₂(t)=+arccos c(t)　[리마송 아래 해]')
    e('θ₂(t)=-arccos c(t)　[사이클로이드 위 해]')
    e('θ₁(t)=atan2(y(t),x(t))-atan2(L₂sin θ₂(t),L₁+L₂cos θ₂(t))')
    p('L₁=5.48532, L₂=6.13245. 각도는 라디안이고, 주값 계산 후 ±2π 점프를 보정하여 연속성을 유지한다.',note)
    h('합성함수의 미분')
    e('s=t/T,　q(t)=10s³-15s⁴+6s⁵,　T=3')
    e('q′=(30s²-60s³+30s⁴)/T')
    e('q″=(60s-180s²+120s³)/T²,　q‴=(60-360s+360s²)/T³')
    e('θᵢ(t)=Θᵢ(q(t))')
    e('θ̇ᵢ=Θᵢ′q′')
    e('θ̈ᵢ=Θᵢ″(q′)²+Θᵢ′q″')
    e('θᵢ‴=Θᵢ‴(q′)³+3Θᵢ″q′q″+Θᵢ′q‴')
    p('Θ의 미분은 u에 대한 미분이며 q와 θ의 미분은 시간에 대한 미분이다. Θ와 그 도함수는 u=q(t)에서 평가한다. 실제 코드는 해석적 미분기구학과 적응형 적분을 사용한다.',note)
    h('세 가지 평가함수')
    e('E<sub>θ</sub>=∫<sub>0</sub><super>3</super> (θ̇₁²+θ̇₂²) dt')
    e('A<sub>θ</sub>=∫<sub>0</sub><super>3</super> (θ̈₁²+θ̈₂²) dt')
    e('J<sub>θ</sub>=∫<sub>0</sub><super>3</super> [(θ₁‴)²+(θ₂‴)²] dt')
    table(['평가값','리마송','사이클로이드'],values([
        ['Eθ [rad²/s]','joint_velocity_effort'],['Aθ [rad²/s³]','joint_acceleration_effort'],['Jθ [rad²/s⁵]','joint_jerk_effort']]),[171,170,170])
    p('이 합성식 전체의 초등함수 부정적분은 제시하지 않는다. 적분은 수치적으로 계산했다. 저크 비용은 저크의 제곱 적분이지 저크가 일정한지 측정하는 분산은 아니다. Eθ는 에너지가 아니다.',note)

    page('기계적 일과 모형 에너지')
    h('토크와 순 기계적 일')
    e('τ=M(θ)θ̈+C(θ,θ̇)+G(θ)+bθ̇,　b=0.03')
    p('M은 2관절 관성행렬, C는 속도에 따른 결합항, G는 중력항이다. 길이는 1 unit=0.1 m로 바꾸고 선밀도 0.35 kg/m의 균일 막대 모형을 쓴다. 토크는 N·m 단위다.',note)
    e('W<sub>net</sub>=∫<sub>0</sub><super>3</super> (τ₁θ̇₁+τ₂θ̇₂) dt')
    e('W<sub>net</sub>=Δ(K+U)+b∫<sub>0</sub><super>3</super> (θ̇₁²+θ̇₂²) dt')
    p('이는 손계산의 일-에너지 관계와 연결된다. 순 일이 음수여도 소비 전기에너지가 음수라는 뜻은 아니다. 두 후보는 팔꿈치 해가 달라 초기·최종 링크 자세와 위치에너지 변화도 다르다.',note)
    h('현재 에너지 평가식')
    e('W<sub>+</sub>=∫<sub>0</sub><super>3</super> [max(τ₁θ̇₁,0)+max(τ₂θ̇₂,0)] dt')
    e('E<sub>copper</sub>=(R/K<sub>t</sub>²)∫<sub>0</sub><super>3</super> (τ₁²+τ₂²) dt')
    e('E<sub>model</sub>=W<sub>+</sub>+E<sub>copper</sub>')
    e('τ<sub>RMS</sub>=√[(1/3)∫<sub>0</sub><super>3</super> (τ₁²+τ₂²) dt]')
    table(['물리량','리마송','사이클로이드'],values([
        ['순 기계적 일 [J]','net_mechanical_work_j'],['양의 기계적 일 [J]','positive_mechanical_work_j'],
        ['모터 동손 [J]','motor_copper_loss_j'],['모형 에너지 [J]','model_electrical_energy_j'],
        ['RMS 토크 [N·m]','rms_joint_torque_nm']]),[171,170,170])
    p('R=0.8 Ω, Kt=0.12 N·m/A는 특정 모터의 실측값이 아니라 가정값이다. 회생 에너지 회수·기어·구동회로 등의 전체 손실은 반영하지 않는다. 실제 장치 소비전력을 검증한 결과로 해석하면 안 된다.',note)
    p('이번 두 후보는 모터 동손이 모형 에너지의 대부분을 차지한다. 따라서 에너지 우열을 보편적인 결과로 주장하지 말고 R과 Kt 등 가정의 영향을 명시해야 한다. 물리 적분값은 641점 사다리꼴 계산이며 161/321/641점 비교로 수치 검사를 했다.',note)

    page('장애물과 종합점수와 결과 해석')
    h('장애물 접근 벌점')
    e('O=∫<sub>0</sub><super>3</super> 1/[max(d(t),δ)]² dt,　δ=0.05 unit')
    p('d(t)는 두 링크 전체와 장애물 사이의 부호 있는 여유다. 링크 반지름 0.05 unit를 반영한다. 끝점과 장애물 사이 거리만 사용하는 것이 아니다. 충돌 후보는 점수 비교 전에 제외한다.')
    p('이 식은 전체 이동의 접근 정도를 누적한다. 최소 여유만 보는 지표와 달라서, O가 작다고 최소 여유가 반드시 큰 것은 아니다. 저장된 최소 여유는 표본 검사에 근거한 값이며 연속 구간의 엄밀한 최솟값 보증은 아니다.',note)
    h('종합점수')
    e('n<sub>k</sub>=(f<sub>k</sub>-min f<sub>k</sub>)/(max f<sub>k</sub>-min f<sub>k</sub>)')
    e('C=100(0.25n<sub>L</sub>+0.20n<sub>E</sub>+0.20n<sub>J</sub>+0.20n<sub>Kε</sub>+0.15n<sub>O</sub>)')
    e('종합점수=100-C')
    p('min과 max는 두 곡선만이 아니라 같은 상황의 적격 후보 12개에서 구한다. 범위가 0인 항은 n=0으로 처리한다. 저장된 기존 가중치·점수·라벨을 유지했으며, 정밀 검산값으로 일부 항만 교체해 재채점하지 않았다.',note)
    table(['평가 항목','리마송 아래','사이클로이드 위'],values([
        ['길이 원자료 [unit]','path_length_units'],['관절 노력도 [rad²/s]','joint_velocity_effort'],
        ['각가속도 비용 [rad²/s³]','joint_acceleration_effort'],['저크 비용 [rad²/s⁵]','joint_jerk_effort'],
        ['정칙화 곡률 [1/unit]','regularized_curvature_energy_inv_unit'],['접근 벌점 [s/unit²]','obstacle_penalty_s_per_unit2'],
        ['최소 장애물 여유 [unit]','min_obstacle_clearance_units'],['모형 에너지 [J]','model_electrical_energy_j'],['종합점수 [점]','score']]),[171,170,170])
    p('이 두 후보끼리는 사이클로이드가 길이·관절 운동 비용·모형 에너지·종합점수에서 낮은 비용 또는 높은 점수를 얻었다. 리마송은 접근 벌점이 더 작았다. 종합 우승 후보는 전체 42개 중 원호 위 해이며, 이 표의 두 후보 중 우승과 구분한다.')
    p('두 후보는 곡선 종류뿐 아니라 팔꿈치 해도 다르다. 차이를 곡선 종류 하나의 효과로 단정하지 않는다. 신뢰성 플래그는 수치 선별 결과이며 엄밀한 오차 보증은 아니다. 원자료·기존 그래프는 보존했다.',note)
    p('재현 기록: 전체평가_검산.json. 원자료: 분석자료_1의 candidate_metrics.csv 및 robot_path_run.json. 수치 검산은 고정 입력으로 재실행했으며 난수나 머신러닝 학습은 사용하지 않았다.',note)
    path=OUT/'리마송_사이클로이드_전체평가_풀이.pdf'
    def footer(c,d):
        c.setFont('Malgun',9);c.drawRightString(A4[0]-42,22,str(d.page))
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=34,bottomMargin=34,title='리마송과 사이클로이드 전체 평가 풀이',author='Robot Path Lab')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    pages=PdfReader(path).pages
    assert len(pages)==5,len(pages)
    assert '-4.8' in pages[0].extract_text()
    assert '285.43791805' in pages[3].extract_text()
    print(f'PDF created: {path} ({len(pages)} pages)')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--compute',action='store_true');ap.add_argument('--pdf',action='store_true');a=ap.parse_args()
    if a.compute:compute()
    if a.pdf:pdf()
