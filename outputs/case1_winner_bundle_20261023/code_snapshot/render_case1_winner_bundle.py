"""Readable PDF/TXT dossier for all six winners, including curvature co-winner."""
from pathlib import Path
import argparse,html,json,math,re

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'outputs/case1_winner_bundle_20261023'

def main(out):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from PIL import Image as PILImage
    from pypdf import PdfReader
    data=json.loads((out/'dossier.json').read_text(encoding='utf-8'))
    pdfmetrics.registerFont(TTFont('Malgun','C:/Windows/Fonts/malgun.ttf'))
    pdfmetrics.registerFont(TTFont('MalgunBold','C:/Windows/Fonts/malgunbd.ttf'))
    body=ParagraphStyle('body',fontName='Malgun',fontSize=10.2,leading=15.3,spaceAfter=6)
    equation=ParagraphStyle('equation',parent=body,fontSize=10.2,leading=17,leftIndent=6,spaceAfter=3)
    title=ParagraphStyle('title',fontName='MalgunBold',fontSize=17,leading=23,spaceAfter=10)
    heading=ParagraphStyle('heading',fontName='MalgunBold',fontSize=11.5,leading=17,spaceBefore=10,spaceAfter=5)
    small=ParagraphStyle('small',parent=body,fontSize=8.7,leading=12.4,textColor=HexColor('#475569'),spaceAfter=4)
    th=ParagraphStyle('th',parent=small,fontName='MalgunBold',textColor=HexColor('#172033'))
    story=[];txt=[];page_no=0;page_titles=[]
    def norm(t):
        t=str(t).replace('−','-').replace('‴',"'''").replace('≈','~')
        for char,number in [('₀','0'),('₁','1'),('₂','2'),('ᵢ','i'),('ⱼ','j'),('ₖ','k'),('₊','+')]:t=t.replace(char,f'<sub>{number}</sub>')
        for char,number in [('⁰','0'),('¹','1'),('²','2'),('³','3'),('⁴','4'),('⁵','5'),('⁶','6'),('⁷','7'),('⁸','8'),('⁹','9'),('⁻','-')]:t=t.replace(char,f'<super>{number}</super>')
        return t
    def plain(t):
        t=re.sub('<sub>(.*?)</sub>',r'_\1',str(t));t=re.sub('<super>(.*?)</super>',r'^(\1)',t)
        return html.unescape(re.sub('<[^>]+>','',t))
    def p(t,style=body):story.append(Paragraph(norm(t),style));txt.append(plain(t))
    def e(t):p(t,equation)
    def h(t):p(t,heading)
    def page(t):
        nonlocal page_no
        if story:story.append(PageBreak())
        page_no+=1;page_titles.append(t);txt.extend(['',f'===== {page_no}쪽 {t} =====','']);p(t,title)
    def table(headers,rows,widths,style=small):
        txt.append('\t'.join(plain(x) for x in headers))
        txt.extend('\t'.join(plain(x) for x in r) for r in rows)
        cells=[[Paragraph(norm(x),th if i==0 else style) for x in row] for i,row in enumerate([headers]+rows)]
        tab=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.45,HexColor('#D9D9D9')),
            ('BACKGROUND',(0,0),(-1,0),HexColor('#E8EDF5')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
        story.extend([tab,Spacer(1,6)])
    def figure(path,maxw=470,maxh=220):
        with PILImage.open(path) as im:w,h=im.size
        factor=min(maxw/w,maxh/h);story.append(Image(str(path),width=w*factor,height=h*factor))
        txt.append('그림 파일: '+str(path.relative_to(out)))
    def fmt(v):return f'{float(v):.9g}'
    names={'path_length_units':'L','model_electrical_energy_j':'E_model','joint_jerk_effort':'J',
           'regularized_curvature_energy_inv_unit':'Kε','obstacle_penalty_s_per_unit2':'O'}
    labels={'distance':'거리','joint_effort':'관절 노력도','acceleration':'각가속도','jerk':'관절 저크',
            'curvature':'정칙화 곡률','obstacle_penalty':'접근 벌점','model_energy':'모형 에너지',
            'rms_torque':'RMS 토크','report_balanced':'종합 비용'}
    short=lambda rid:rid.replace('interactive_','')
    items=data['items'];byid={i['run_id']:i for i in items}

    page('분석자료 1 우승 곡선 전체 자료')
    p('기존 분석자료_1의 같은 조건을 기준으로, 9개 평가목적의 우승 후보와 공동 우승 후보를 모았다. 후보 42개 중 12개가 검사를 통과했고, 중복을 제거하면 우승 후보는 6개다.')
    p('로봇 밑부분 (0, 0), L₁=5.48532 unit, L₂=6.13245 unit. 끝점은 (2, 0)에서 (-4.8, -1.5)까지 3초 동안 이동한다. 장애물은 중심 (-1.3, -2), 반지름 0.45 unit인 원 하나다.')
    table(['평가기준','우승 후보','최솟값','동점 수'],[
        [labels[r['objective']],'<br/>'.join(short(x) for x in json.loads(r['tied_run_ids_json'])),fmt(r['raw_metric_value']),str(int(r['tie_count']))]
        for r in data['labels']],[91,245,112,63])
    p('개별 평가값은 최소화하는 비용이다. 종합 라벨의 값 2.9952487147도 비용이며, 화면 종합점수는 100-비용=97.0047512853점이다. 개별 항목에 존재하지 않는 별도 100점 점수를 새로 만들지 않았다.',small)
    p('곡률 공동 우승은 bezier_p80_up과 bezier_m80_down이다. 대표 표시 후보는 저장된 라벨대로 bezier_p80_up이며, 이 자료에는 두 후보 모두 포함했다. 동점은 상대·절대 허용오차 10⁻¹²로 판단한다. 대표는 원시 비용, 길이, 곡선 키, 굽힘값, 팔꿈치 해, ID 순의 기존 결정 규칙을 따른다.',small)
    h('자료를 읽는 순서')
    p('2~3쪽은 공통 매개변수·역기구학·적분식이다. 4~15쪽은 후보별로 식과 구간 및 그래프 1쪽, 평가 적분과 점수 계산 1쪽씩 배치했다. 16~17쪽에는 정밀 검산과 물리 모형을 정리했다.')
    p('모든 최적은 준비된 42개 안에서의 최적이다. 모형 에너지는 가정한 로봇팔·모터의 계산값이며 실측 소비전력이 아니다. 원자료와 기존 점수는 보존하고, 정밀 검산값은 별도로 표시했다.',small)

    page('매개변수와 관절각의 공통 식')
    h('어디에서 어디까지 움직이는가')
    e('모든 후보: u=0 → 1,　t=0 → 3 s,　p(0)=(2,0),　p(1)=(-4.8,-1.5)')
    p('u는 경로의 진행률이며 x좌표 자체가 아니다. 특히 리마송은 x가 단조롭게 움직이지 않으므로 x=-4.8부터2인 단일 함수 y=f(x)로 제한해서 읽으면 안 된다. 각 후보의 u 또는 각도 구간을 따라 곡선을 읽는다.')
    e('s=t/3,　u=q(t)=10s³-15s⁴+6s⁵')
    e('q′=(30s²-60s³+30s⁴)/3')
    e('q″=(60s-180s²+120s³)/9')
    e("q'''=(60-360s+360s²)/27")
    h('끝점 좌표에서 관절각으로 변환')
    e('c(t)=[x(t)²+y(t)²-5.48532²-6.13245²]/[2×5.48532×6.13245]')
    e('θ₂(t)=-arccos c(t) [위 해],　θ₂(t)=+arccos c(t) [아래 해]')
    e('θ₁(t)=atan2(y(t),x(t))')
    e('　　　 -atan2(6.13245 sin θ₂(t),5.48532+6.13245 cos θ₂(t))')
    p('x(t),y(t)는 각 후보의 좌표식에 u=q(t)를 대입한 것이다. 각도는 라디안이다. 주값 계산 후 시간에 따른 ±2π 점프를 보정하여 연속성을 유지한다. 위·아래는 끝점 곡선의 위치가 아니라 두 역기구학 해의 이름이다.',small)
    h('관절각의 시간 미분')
    e('θᵢ(t)=Θᵢ(q(t))')
    e('θᵢ′=Θᵢ′q′')
    e('θᵢ″=Θᵢ″(q′)²+Θᵢ′q″')
    e("θᵢ'''=Θᵢ'''(q′)³+3Θᵢ″q′q″+Θᵢ′q'''")
    p('Θ의 미분은 u에 대한 미분, θ와 q의 미분은 t에 대한 미분이다. 모든 Θ 도함수는 u=q(t)에서 평가한다. 실제 계산에는 해석적 미분기구학을 사용했다. θ(t)를 단순 다항식으로 임의 근사하거나, 다른 곡선식으로 바꾸지 않았다.',small)
    p('정밀한 전체 합성식과 파라미터는 data/우승후보_정밀원본식.csv에 있다. 본문 계수는 표시용 반올림이다. 후보마다 같은 식을 불필요하게 길게 반복하는 대신 이 공통식과 후보별 x,y 및 팔꿈치 해를 함께 사용한다.',small)

    page('실제로 적분한 평가함수')
    e('L=∫₀¹√(x′²+y′²) du')
    e('Eθ=∫₀³[(θ₁′)²+(θ₂′)²]dt,　Aθ=∫₀³[(θ₁″)²+(θ₂″)²]dt')
    e("Jθ=∫₀³[(θ₁''')²+(θ₂''')²]dt")
    e('Kε=∫₀¹ (x′y″-y′x″)²/(x′²+y′²+ε²)<super>5/2</super> du')
    e('ε=0.001√[(-4.8-2)²+(-1.5-0)²]=0.006963476143421474')
    p('L,Kε의 미분은 u에 대한 미분이다. Eθ,Aθ,Jθ의 미분은 시간 t에 대한 미분이다. 저크 비용은 각저크 제곱 적분으로, 저크 일정성이나 곡률 자체와는 다르다.',small)
    e('O=∫₀³ 1/[max(d(t),0.05)]² dt')
    p('d(t)는 모든 장애물과 두 링크 선분 사이의 최소 중심거리에서 장애물 반지름 및 링크 반지름 0.05 unit를 뺀 부호 있는 여유다. 벌점 하한 0.05와 링크 반지름 0.05는 역할이 다른 설정이다. 충돌 후보는 별도로 제외한다.',small)
    e('W_net=∫₀³(τ₁θ₁′+τ₂θ₂′)dt')
    e('W₊=∫₀³[max(τ₁θ₁′,0)+max(τ₂θ₂′,0)]dt')
    e('E_copper=(0.8/0.12²)∫₀³(τ₁²+τ₂²)dt,　E_model=W₊+E_copper')
    e('τ_RMS=√[(1/3)∫₀³(τ₁²+τ₂²)dt]')
    h('수치적분 과정의 기록')
    p('L과 Kε는 uᵢ=(1-cos(πi/160))/2인 161개 기하 표본에서 계산했다. O는 161개 균일 시간 표본을 사용한다. 각 구간 기여값은 Δzᵢ(fᵢ+fᵢ₊₁)/2이고, 전체 합이 저장된 평가값이다.')
    p('Eθ,Aθ,Jθ는 적응형 QUADPACK 분할을 이용해 32점 가우스-르장드르 적분으로 재현했다. 패널 j와 노드 k에서 계산한 가중치 wⱼₖ와 피적분값 fⱼₖ의 곱을 모두 더한다. 실제 노드·가중치·기여값 전체를 CSV로 첨부했다.')
    p('물리량은 641개 균일 시간 표본에서 사다리꼴법으로 적분했다. RMS는 토크 제곱 적분을 3초로 나누고 제곱근을 취한다. 641점은 정확한 참값이 아니라 사용한 계산 해상도다.',small)

    for item in items:
        rid=item['short_id'];m=item['metrics'];ad=item['adaptive'];ph=item['physics_integrals']
        page(f"{item['index']} {item['display']} 곡선식과 구간")
        p(rid+' · '+' / '.join(item['reasons']),small)
        p('시작 (2,0) → 도착 (-4.8,-1.5). '+' / '.join(item['domains']),small)
        for line in item['equations_display']:e(line)
        if item['curve_key']=='bezier':p('대칭 제어점으로 3차항이 소거되는 후보다. 10⁻¹² 미만의 부동소수점 잔차는 표시에서 생략했다.',small)
        if item['curve_key']=='limacon':p('r의 극점은 로봇 밑부분이 아니다. 위 식은 확대·회전·평행이동된 리마송이다. φ(t)=-3π/5+(6π/5)q(t).',small)
        if item['curve_key']=='cycloid':p('X,Y는 곡선의 국소 좌표다. A와 B가 다른 일반화된 사이클로이드이며 φ(t)=2πq(t).',small)
        figure(out/item['figure'],maxh=215)
        p('화살표는 진행 방향이다. 끝점 경로만 그렸으며 특정 후보의 로봇팔 자세는 겹쳐 그리지 않았다. 그림마다 축 범위는 다를 수 있지만 x,y 축척은 같다.',small)
        p('좌표의 미분 기준: '+item['derivative_variable'],small)
        for line in item['derivatives']:e(line)
        p('길이 검산: '+item['length_recipe'],small)
        p('길이 검산값 '+fmt(item['length_reference'])+' unit (저장된 161점 근삿값과 구분).',small)
        a=item['end_angles']['start'];b=item['end_angles']['end']
        p(f"관절각 시작 (θ₁,θ₂)=({a['theta1_rad']:.7f},{a['theta2_rad']:.7f}) rad, 도착 ({b['theta1_rad']:.7f},{b['theta2_rad']:.7f}) rad. 시간별 θ는 공통 역기구학 식으로 정의한다.",small)

        page(f"{item['index']} {item['display']} 적분과 점수")
        p(rid+' · 원자료와 같은 입력·팔꿈치 해·이동시간으로 검산했다.',small)
        process=[
          ['L [unit]','161개 u 표본의 160개 사다리꼴 합',fmt(m['path_length_units'])],
        ]
        for sym,field,unit in [('Eθ','joint_velocity_effort','rad²/s'),('Aθ','joint_acceleration_effort','rad²/s³'),('Jθ','joint_jerk_effort','rad²/s⁵')]:
            details=ad[field];process.append([f'{sym} [{unit}]',f"{details['panels']}개 패널 ×32노드<br/>Σwⱼₖfⱼₖ ({details['nodes']}항)",fmt(m[field])])
        process.extend([
          ['Kε [1/unit]','161개 u 표본의 160개 사다리꼴 합',fmt(m['regularized_curvature_energy_inv_unit'])],
          ['O [s/unit²]','161개 t 표본의 160개 사다리꼴 합',fmt(m['obstacle_penalty_s_per_unit2'])],
          ['E_model [J]',f"W₊ {fmt(item['positive_work'])}<br/>+동손 {fmt(item['copper'])}",fmt(m['model_electrical_energy_j'])],
          ['τ_RMS [N·m]',f"sqrt({fmt(ph['torque_squared_nm2'])}/3)",fmt(m['rms_joint_torque_nm'])],
          ['W_net [J] 참고','641개 t 표본의 640개 사다리꼴 합',fmt(item['net_work'])],
        ])
        table(['평가량','대입과 누적 과정','계산 결과'],process,[120,265,126])
        p('각 구간의 양 끝 피적분값·간격·기여값·누적합과 관절 적응형 노드 전체는 integration_steps 폴더에 있다. 이 표는 그 합을 요약한 것이다. 적분의 초등함수 부정적분을 모두 구했다는 뜻은 아니다.',small)
        h('종합점수의 정규화와 가중합')
        scoretable=[[names[x['metric']],fmt(x['raw_value']),fmt(x['minimum']),fmt(x['maximum']),f"{x['normalized']:.6f}",f"{100*x['weight']:.0f}%",f"{x['deduction_points']:.6f}"] for x in item['scores']]
        table(['항목','원시값','최솟값','최댓값','정규화 n','가중치','차감점'],scoretable,[45,86,82,86,73,57,82])
        e('정규화 n=(원시값-최솟값)/(최댓값-최솟값)')
        e('C='+ '+'.join(f"{x['deduction_points']:.6f}" for x in item['scores'])+f" ~ {m['report_balanced_cost_0_100']:.8f}")
        e(f"종합점수=100-C ~ {item['score']:.8f}점")
        p('min/max는 이 6개가 아니라 적격 후보 12개를 기준으로 한다. 표는 반올림했으며 실제 계산은 첨부 CSV의 정밀값을 쓴다. 범위가 0인 항은 n=0이다. 높은 점수가 좋으며, 비용·점수는 다른 상황과 직접 비교하지 않는다.',small)
        p('관절 노력도·각가속도·RMS 토크는 개별 목적이지만 현재 종합 가중합에는 직접 들어가지 않는다. 8개 피적분함수 그래프: '+item['integrands_figure'],small)

    page('길이와 정칙화 곡률의 정밀 검산')
    p('정밀 검산값은 저장된 근삿값을 덮어쓰지 않는다. 기존 라벨과 점수는 원자료 값으로 계산했다. 부분적으로 더 정밀한 값만 넣어 기존 점수와 섞지 않았다.')
    rows=[]
    for item in items:
        rows.append([item['short_id'],fmt(item['metrics']['path_length_units']),fmt(item['length_reference']),
                     fmt(item['metrics']['regularized_curvature_energy_inv_unit']),fmt(item['Kepsilon_reference'])])
    table(['후보','저장 길이','길이 검산','저장 Kε','Kε 검산'],rows,[166,86,86,86,87])
    p('길이 단위는 unit, Kε 단위는 1/unit이다. 원호·베지어·사이클로이드 길이는 해석식, 에르미트·리마송 길이는 독립 적응형 적분으로 검산했다. Kε는 같은 u와 같은 ε로 독립 적응형 적분했다. 이는 비교용 수치 기준이며 엄밀한 오차 상한의 증명은 아니다.',small)
    h('부정적분으로 정리하는 길이')
    e('원호: L=R|Δα|')
    e('베지어: 속도²=au²+bu+c,　z=u+b/(2a),　k²=(4ac-b²)/(4a²)')
    e('∫√(au²+bu+c)du')
    e('　=(√a/2)[z√(z²+k²)+k²ln((z+√(z²+k²))/k)]+상수')
    e('사이클로이드: z=cos(φ/2),　L=8∫₀¹√(A²+c²z²)dz,　c=√(B²-A²)')
    e('　L=4[B+(A²/c)ln((B+c)/A)]')
    h('곡률의 뜻과 뾰족점')
    e('κ=|x′y″-y′x″|/(x′²+y′²)<super>3/2</super>')
    p('리마송은 κ=|2.2482+2.37cosφ|/[s(1.6241+1.58cosφ)^(3/2)]이다. 사이클로이드는 κ=AB(1-cosφ)/[A²(1-cosφ)²+B²sin²φ]^(3/2)이며 두 끝점에서는 정의되지 않는다.',small)
    p('사이클로이드 끝점에서 κ~A/(2B²φ), ds~Bφdφ이므로 κ²ds는 상수×dφ/φ에 비례하여 발산한다. 따라서 보통 제곱 곡률 적분 ∫κ²ds와 유한한 정칙화 비용 Kε를 혼동하지 않는다. φ로 치환할 때 속도 정칙화값도 ε/(dφ/du)로 바꿔야 한다.',small)

    page('토크 모형과 가정 및 검증 범위')
    p('두 링크를 균일한 막대, 두 관절을 직접 구동하는 등가 모터로 가정했다. 아래 식의 ℓ은 SI 길이이며 원자료의 unit 길이와 구분한다.')
    e('ℓᵢ=0.1Lᵢ [m],　mᵢ=0.35ℓᵢ [kg],　cᵢ=ℓᵢ/2,　Iᵢ=mᵢℓᵢ²/12')
    e('τ₁=M₁₁θ₁″+M₁₂θ₂″+C₁+G₁+bθ₁′')
    e('τ₂=M₁₂θ₁″+M₂₂θ₂″+C₂+G₂+bθ₂′')
    e('M₁₁=I₁+I₂+m₁c₁²+m₂(ℓ₁²+c₂²+2ℓ₁c₂cosθ₂)')
    e('M₁₂=I₂+m₂(c₂²+ℓ₁c₂cosθ₂),　M₂₂=I₂+m₂c₂²')
    e('C₁=-m₂ℓ₁c₂sinθ₂(2θ₁′θ₂′+θ₂′²)')
    e('C₂=m₂ℓ₁c₂sinθ₂θ₁′²')
    e('G₁=g[(m₁c₁+m₂ℓ₁)cosθ₁+m₂c₂cos(θ₁+θ₂)]')
    e('G₂=gm₂c₂cos(θ₁+θ₂)')
    p('g=9.80665 m/s², b=0.03 N·m·s/rad, R=0.8 Ω, Kt=0.12 N·m/A.',small)
    p('R, Kt, 선밀도, 감쇠는 특정 모터·장치를 측정한 값이 아니라 설정한 가정값이다. 에너지의 절대값이나 우승 순위는 가정에 의존한다. 회생제동, 기어와 구동회로의 전체 손실 및 모터 전류·토크 한계를 모두 검증한 모형은 아니다.',small)
    h('기계적 일과 검산')
    e('W_net=Δ(운동에너지+위치에너지)+bEθ')
    p('동일한 시작·도착 끝점이어도 위·아래 해가 다르면 링크 자세와 위치에너지 변화가 다를 수 있다. 순 일이 음수라고 소비 에너지가 음수인 것은 아니다. 서로 다른 곡선·팔꿈치 해의 차이를 곡선 종류 하나의 효과로 단정하지 않는다.')
    h('수행한 검증')
    p('원자료 42개의 유효성·평가값과 9개 승자 라벨을 재실행해 대조했다. 선택된 6개의 L/Kε/O 구간합, Eθ/Aθ/Jθ 적응형 가중합, 641점 물리 적분, 정규화와 종합점수를 각각 재구성해 원자료와 대조했다. 곡선의 u=0과1 좌표를 시작·도착점과 확인하고 원본 파일 해시가 바뀌지 않았는지 확인했다.')
    p('유효·수치검사 통과는 특정 해상도와 모형에서의 선별 결과이지 완벽한 물리 안전성이나 엄밀한 정확도 보증이 아니다. 저장된 최소 장애물 여유 역시 표본 검사에 근거한 값이다. 이 자료에서는 새로운 가중치 조정, 데이터 재생성, 머신러닝 학습, 통계적 유의성 주장을 하지 않았다.',small)
    p('입력: 분석자료_1/robot_path_run.json 및 같은 폴더의 비교 CSV. 계산 버전: source phase4-mathematics-v3 + report-v4 물리 모형. 전체 원본과 소스 스냅샷 및 해시는 묶음 파일에 포함했다. 난수 추출은 사용하지 않았다.',small)
    pdfpath=out/'우승곡선_전체자료.pdf'
    def footer(c,d):
        c.setFont('Malgun',9);c.setFillColor(HexColor('#64748b'));c.drawRightString(A4[0]-40,22,str(d.page))
    doc=SimpleDocTemplate(str(pdfpath),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=31,bottomMargin=33,title='분석자료 1 우승 곡선 전체 자료',author='Robot Path Lab')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(pdfpath)
    assert len(reader.pages)==17,(len(reader.pages),page_titles)
    for n,expected in enumerate(page_titles):
        content=reader.pages[n].extract_text()
        assert plain(expected) in content,(n,expected)
        assert '\ufffd' not in content
    assert '-4.8' in reader.pages[1].extract_text()
    (out/'우승곡선_전체자료.txt').write_text('\n\n'.join(txt),encoding='utf-8-sig')
    (out/'pdf_checks.json').write_text(json.dumps(dict(pages=17,page_titles_checked=True,all_six_candidates=True,co_winner_included=True,
        text_extraction_passed=True,visual_review_pending=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Created {len(reader.pages)} pages: {pdfpath}')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT_OUT);args=ap.parse_args();main(args.output)
