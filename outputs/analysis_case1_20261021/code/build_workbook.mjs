import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const out=path.resolve(process.argv[2] ?? '../../outputs/analysis_case1_20261021');
const data=JSON.parse(await fs.readFile(path.join(out,'workbook_payload.json'),'utf8'));
const wb=Workbook.create();
const summary=wb.worksheets.add('후보 비교');
const raw=wb.worksheets.add('42개 후보 원자료');
const settings=wb.worksheets.add('실험 조건');
const font='Malgun Gothic',ink='#172033',blue='#3563e9';
function base(s,range){
  s.showGridLines=false;
  s.getRange(range).format.font={name:font,size:10,color:ink};
  s.getRange(range).format.rowHeight=23;
  s.getRange(range).format.columnWidth=17;
  s.getRange(range).format.verticalAlignment='center';
}
function header(s,range){s.getRange(range).format={fill:'#26334d',font:{name:font,size:10,bold:true,color:'#ffffff'},wrapText:true,horizontalAlignment:'center',rowHeight:36};}
base(summary,'A1:J54');base(raw,'A1:S65');base(settings,'A1:B45');
summary.tabColor='#26334d';
summary.getRange('A2').values=[['분석자료 1 · 평가기준별 후보 비교']];summary.getRange('A2').format.font={name:font,size:15,bold:true};
summary.getRange('A4').values=[['동일 입력의 42개 중 유효 12개. 목적별 우승 후보를 중복 제거한 5개.']];
summary.getRange('A6:I6').values=[data.summary_headers];header(summary,'A6:I6');
summary.getRange('A7:B11').values=data.summary_rows.map(r=>r.slice(0,2));
const sourceRows=new Map(data.all_rows.map((r,i)=>[r[0],i+4]));
const {selected_run_ids:summaryIds}=JSON.parse(await fs.readFile(path.join(out,'experiment_metadata.json'),'utf8'));
const sourceCols=['J','Q','K','M','N','O','S'];
for(let i=0;i<5;i++) summary.getRange(`C${i+7}:I${i+7}`).formulas=[sourceCols.map(c=>`='42개 후보 원자료'!${c}${sourceRows.get(summaryIds[i])}`)];
summary.getRange('C7:I11').setNumberFormat('#,##0.000');
summary.getRange('A7:B11').format.wrapText=true;summary.getRange('A7:I11').format.rowHeight=40;
summary.getRange('A6:A25').format.columnWidth=18;summary.getRange('B6:B25').format.columnWidth=33;
summary.getRange('C6:I25').format.columnWidth=19;
summary.getRange('A14').values=[['평가기준별 승자 라벨 (검증된 계산 결과의 스냅샷)']];
summary.getRange('A16:E16').values=[data.winner_headers];header(summary,'A16:E16');
summary.getRange('A17:E25').values=data.winner_rows;
summary.getRange('B17:B25').format.columnWidth=33;
summary.getRange('C17:C25').setNumberFormat('#,##0.000000');summary.getRange('D17:D25').setNumberFormat('0');
summary.getRange('D17:E25').format.horizontalAlignment='center';
summary.getRange('A27').values=[['종합 라벨의 값은 비용(작을수록 좋음). 위 비교표는 100−비용인 점수(클수록 좋음).']];

raw.getRange('A1').values=[['저장된 42개 후보 · 제외 후보와 결측값도 보존']];
raw.getRange('A3:S3').values=[data.all_headers];header(raw,'A3:S3');raw.getRange('A3:S3').format.rowHeight=55;
raw.getRange('A4:S45').values=data.all_rows;raw.getRange('A4:A45').format.columnWidth=38;
raw.getRange('B4:I45').format.columnWidth=16;raw.getRange('J4:S45').setNumberFormat('0.000000');
raw.freezePanes.freezeRows(3);
raw.tables.add('A3:S45',true,'AllCandidates');
raw.getRange('A49').values=[['유효 후보의 그래프 좌표 (위 원자료에 연결)']];
raw.getRange('A50:C50').values=[['후보','길이 [unit]','모형 에너지 [J]']];header(raw,'A50:C50');
let helper=51;
for(let i=0;i<data.all_rows.length;i++){
  const r=data.all_rows[i];if(r[5]!==true || r[6]!==true)continue;
  raw.getRange(`A${helper}`).formulas=[[`=A${i+4}`]];
  raw.getRange(`B${helper}:C${helper}`).formulas=[[`=J${i+4}`,`=Q${i+4}`]];
  helper++;
}
raw.getRange('B51:C62').setNumberFormat('0.000000');

const bar=summary.charts.add('bar',[summary.getRange('A6:A11'),summary.getRange('C6:C11')]);
bar.title='선택 후보 5개의 경로 길이';bar.hasLegend=false;bar.setPosition('A30','E48');
bar.xAxis={axisType:'textAxis',textStyle:{typeface:font,fontSize:11}};
bar.yAxis={numberFormatCode:'0.0',numberFormatSourceLinked:false,textStyle:{typeface:font,fontSize:11}};
bar.yAxis.title.text='경로 길이 [unit]';bar.series.items[0].fill=blue;
const scatter=summary.charts.add('scatter',[raw.getRange('B50:B62'),raw.getRange('C50:C62')]);
scatter.title='유효 후보 12개의 거리–에너지';
scatter.hasLegend=false;scatter.setPosition('F30','J48');
scatter.xAxis={numberFormatCode:'0.0',numberFormatSourceLinked:false,textStyle:{typeface:font,fontSize:11}};
scatter.yAxis={numberFormatCode:'0',numberFormatSourceLinked:false,textStyle:{typeface:font,fontSize:11}};
scatter.xAxis.title.text='경로 길이 [unit]';scatter.yAxis.title.text='모형 에너지 [J]';scatter.series.items[0].fill='#64748b';
for(const chart of [bar,scatter]){chart.titleTextStyle.typeface=font;chart.titleTextStyle.fontSize=13;}
summary.getRange('A51').values=[['모형 에너지는 실측 전력이 아님. 원자료 값 수정 시 표·차트는 연결되지만 승자 라벨은 재계산되지 않음.']];
summary.getRange('A52').values=[['조건을 바꾸려면 앱에서 다시 비교·저장. 이 사례의 결론은 준비된 42개 후보 안에서만 유효.']];
summary.getRange('A53').values=[['단위: 길이 unit, 에너지 J, 노력도 rad²/s, 저크 비용 rad²/s⁵, Kε 1/unit, 접근 벌점 s/unit².']];
settings.getRange('A2:B2').values=[data.settings[0]];header(settings,'A2:B2');
settings.getRange(`A3:B${data.settings.length+1}`).values=data.settings.slice(1).map(([k,v])=>[k,
  typeof v==='string' && /^-?\d+(\.\d+)?(e[+-]?\d+)?$/i.test(v) ? Number(v) : v]);
settings.getRange('A2:A42').format.columnWidth=39;settings.getRange('B2:B42').format.columnWidth=100;
settings.getRange('B3:B42').format.wrapText=true;settings.getRange('B3:B42').format.rowHeight=30;
settings.getRange(`B${data.settings.length}:B${data.settings.length+1}`).format.rowHeight=60;
wb.recalculate();
const expected=data.summary_rows.map(r=>r.slice(2));
const actual=summary.getRange('C7:I11').values;
for(let i=0;i<5;i++)for(let k=0;k<7;k++)if(Math.abs(actual[i][k]-expected[i][k])>1e-8)throw new Error('Summary linkage mismatch');
// Verify formula-linked metric and chart inputs respond, then restore exactly.
const changeRow=sourceRows.get(summaryIds[0]);
const old=raw.getRange(`J${changeRow}`).values;raw.getRange(`J${changeRow}`).values=[[123.5]];wb.recalculate();
if(summary.getRange('C7').values[0][0]!==123.5)throw new Error('Linked summary update failed');
if(raw.getRange('B51:B62').values.flat().every(v=>v!==123.5))throw new Error('Linked scatter update failed');
raw.getRange(`J${changeRow}`).values=old;wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},maxChars:3000});
console.log(errors.ndjson);
await fs.mkdir(path.join(out,'previews'),{recursive:true});
for(const [sheetName,range,name] of [['후보 비교','A1:I27','summary'],['후보 비교','A29:J54','charts'],['42개 후보 원자료','A1:S45','source'],['실험 조건','A1:B30','settings']]){
  const img=await wb.render({sheetName,range,scale:1.4,format:'png'});
  await fs.writeFile(path.join(out,'previews',name+'.png'),new Uint8Array(await img.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,'사례1_비교표.xlsx'));
await fs.writeFile(path.join(out,'workbook_checks.json'),JSON.stringify({status:'passed',sheets:3,charts:2,selected:5,valid:12,summary_linkage:true,formula_error_scan:errors.ndjson,chart_series:[bar,scatter].map(c=>c.series.items.map(s=>({formula:s.formula,categoryFormula:s.categoryFormula})))},null,2));
console.log('Saved workbook and four rendered previews.');
