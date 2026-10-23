"""Readable PDF and copyable text for the existing team curve handout."""
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/analysis_case1_20261021'
SOURCE=OUT/'팀원용_곡선식.md'
pdfmetrics.registerFont(TTFont('Malgun','C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold','C:/Windows/Fonts/malgunbd.ttf'))
styles={
 'title':ParagraphStyle('title',fontName='MalgunBold',fontSize=18,leading=24,textColor=HexColor('#000000'),spaceAfter=10),
 'h':ParagraphStyle('h',fontName='MalgunBold',fontSize=12,leading=18,spaceBefore=13,spaceAfter=6,textColor=HexColor('#000000')),
 'body':ParagraphStyle('body',fontName='Malgun',fontSize=10.5,leading=16,spaceAfter=6),
 'eq':ParagraphStyle('eq',fontName='Malgun',fontSize=11,leading=18,spaceAfter=3,leftIndent=10),
 'note':ParagraphStyle('note',fontName='Malgun',fontSize=9,leading=14,textColor=HexColor('#475569'),spaceAfter=5),
}
story=[]
def p(text,style='body'):story.append(Paragraph(text.replace('−','-'),styles[style]))
def eq(text):p(text,'eq')
def section(title,lines,note=None):
    group=[Paragraph(title,styles['h'])]+[Paragraph(t.replace('−','-'),styles['eq']) for t in lines]
    if note:group.append(Paragraph(note.replace('−','-'),styles['note']))
    story.append(KeepTogether(group))

p('팀원용 곡선식','title')
p('분석자료 1의 동일한 로봇팔 조건에서 선택된 후보 5개의 식이다. 곡선의 모양과 시간에 따른 이동을 구분해서 읽는다.')
p('시작점 (2, 0), 도착점 (−4.8, −1.5), 링크 길이 5.48532 및 6.13245 unit, 이동시간 3 s. 표시 계수는 반올림했으며 원자료와 평가값은 변경하지 않았다.','note')
section('공통 매개변수와 시간함수',[
 '0 ≤ u ≤ 1,　0 ≤ t ≤ 3 s',
 'u = q(t) = 10(t/3)<super>3</super> − 15(t/3)<super>4</super> + 6(t/3)<super>5</super>',
 'x(t) = x(q(t)),　y(t) = y(q(t))',
], 'u는 곡선의 진행률, t는 시간, φ와 α는 각도다. 삼각함수의 각도는 라디안이다. 아래 x(u), y(u)는 좌표 함수이며 미분 기호가 아니다.')
section('에르미트 아래 해',[
 'x(u) = 2 − 5.36851416262u − 6.65848977295u<super>2</super>',
 '　　　　 + 5.22700393557u<super>3</super>',
 'y(u) = 4.43498091154u − 7.08799648556u<super>2</super>',
 '　　　　 + 1.15301557402u<super>3</super>',
], '후보 ID: hermite_m80_down. 이 상황에서 경로 길이 최소 후보다.')
section('원호 위 해',[
 'α(u) = −0.170313355772 − 2.36674331258u',
 'x(u) = −1.70603571429 + 3.76044291450 cos α(u)',
 'y(u) = 0.637361904762 + 3.76044291450 sin α(u)',
 'L = R |Δα| = 8.90000312021 unit',
], '후보 ID: arc_p80_up. 이 상황에서 종합점수 최대 후보다. 위 길이는 원호의 해석적 길이다.')
section('베지어 위 해',[
 'x(u) = 2 − 5.288u − 1.512u<super>2</super>',
 'y(u) = −8.3544u + 6.8544u<super>2</super>',
], '후보 ID: bezier_p80_up. 정칙화 곡률 비용의 동점 후보 2개 중 기존 규칙으로 선택했다. 대칭 제어점으로 3차항이 소거된다. 10<super>−12</super> 미만의 부동소수점 잔차는 표시에서 생략했다.')
story.append(PageBreak())
p('리마송과 사이클로이드','title')
section('리마송 아래 해',[
 'φ(u) = −3π/5 + (6π/5)u,　−3π/5 ≤ φ ≤ 3π/5',
 'r(φ) = 1 + 0.79 cos φ',
 'x(φ) = −1.64368977217',
 '　　　　 + 4.84327227602 r(φ) cos(φ + 1.78790764153)',
 'y(φ) = 0.354726967192',
 '　　　　 + 4.84327227602 r(φ) sin(φ + 1.78790764153)',
 'L = 4.84327227602 ∫<sub>−3π/5</sub><super>3π/5</super> √(1.6241 + 1.58 cos φ) dφ',
], '후보 ID: limacon_p80_down. 이 상황에서 장애물 접근 벌점 최소 후보다. 극곡선을 확대·회전·평행이동한 식이며 길이는 수치적분한다. 실제 이동은 φ(t) = −3π/5 + (6π/5)q(t)를 대입한다.')
section('사이클로이드 위 해',[
 'φ = 2πu,　0 ≤ φ ≤ 2π',
 'A = 1.10827164932802,　B = 1.39269522868429',
 'X(φ) = A(φ − sin φ),　Y(φ) = B(1 − cos φ)',
 'x(φ) = 2 − 0.976523773464 X(φ) + 0.215409655911 Y(φ)',
 'y(φ) = −0.215409655911 X(φ) − 0.976523773464 Y(φ)',
], '후보 ID: cycloid_p80_up. 이 상황에서 모형 에너지 최소 후보다. A ≠ B인 일반화된 사이클로이드이므로 표준 사이클로이드의 길이 8A를 그대로 사용하면 안 된다. 실제 이동은 φ(t) = 2πq(t)를 대입한다.')
section('두 관절의 각도 합성식',[
 'c(t) = [x(t)<super>2</super> + y(t)<super>2</super> − 5.48532<super>2</super> − 6.13245<super>2</super>]',
 '　　　　 / [2 × 5.48532 × 6.13245]',
 'θ<sub>2</sub>(t) = −arccos c(t)　[위 해]',
 'θ<sub>2</sub>(t) = +arccos c(t)　[아래 해]',
 'θ<sub>1</sub>(t) = atan2(y(t), x(t))',
 '　　　　 − atan2(6.13245 sin θ<sub>2</sub>(t), 5.48532 + 6.13245 cos θ<sub>2</sub>(t))',
], '위·아래는 곡선 위치가 아니라 두 역기구학 해의 이름이다. atan2(y, x)는 두 좌표로 방향각을 구하는 함수다. 프로그램은 주값 계산 후 시간에 따른 ±2π 점프를 보정해 관절각을 연속적으로 연결한다.')
p('곡선별 우승은 이 상황의 준비된 42개 후보 안에서의 결과다. 모형 에너지는 실측 소비전력이 아니다. 정밀한 원본 식과 파라미터는 selected_equations.csv에 보존되어 있다.','note')

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Malgun',9);canvas.setFillColor(HexColor('#64748b'))
    canvas.drawRightString(A4[0]-42,24,f'{doc.page}');canvas.restoreState()

pdf=OUT/'팀원용_곡선식.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=36,bottomMargin=36,
                      title='팀원용 곡선식',author='Robot Path Lab')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
text=SOURCE.read_text(encoding='utf-8')
plain='\n'.join(line.lstrip('#').strip() if line.startswith('#') else line for line in text.splitlines())
(OUT/'팀원용_곡선식.txt').write_text(plain,encoding='utf-8-sig')
reader=PdfReader(pdf)
assert len(reader.pages)==2,len(reader.pages)
assert '리마송' in reader.pages[1].extract_text()
assert '에르미트' in reader.pages[0].extract_text()
assert '-4.8' in reader.pages[0].extract_text()
assert '-3π/5' in reader.pages[1].extract_text()
print(f'Created PDF with {len(reader.pages)} pages and UTF-8 text file.')
