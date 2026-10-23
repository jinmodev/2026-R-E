// Reproducible workbook authoring with the installed @oai/artifact-tool runtime.
// Input: report_phase4.py's typed JSON; no raw time samples or long equations.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'outputs','phase4_analysis');
const payload=JSON.parse(await fs.readFile(path.join(out,'workbook_payload.json'),'utf8'));
const wb=Workbook.create();
const previews=path.join(out,'excel_previews');await fs.mkdir(previews,{recursive:true});
const col=n=>String.fromCharCode(65+n);
for(const p of payload.sheets){
  const sh=wb.worksheets.add(p.name);sh.showGridLines=false;
  if(p.name==='유효성')sh.tabColor='#192840';
  const last=col(p.headers.length-1),end=5+p.rows.length;
  sh.getRange(`A1:${last}${end}`).format.font={name:'Malgun Gothic',size:10,color:'#20314D'};
  sh.getRange(`A1:${last}${end}`).format.rowHeight=25;
  sh.getRange(`A1:${last}${end}`).format.verticalAlignment='center';
  sh.getRange('A2').values=[[p.title]];sh.getRange('A2').format.font={name:'Malgun Gothic',size:14,bold:true,color:'#192840'};
  sh.getRange(`A2:${last}2`).format.borders={bottom:{style:'thin',color:'#9CABBF'}};
  sh.getRange('A3').values=[[p.note]];
  sh.getRange('A3').format.font={name:'Malgun Gothic',size:10,color:'#53627B'};
  sh.getRange(`A5:${last}5`).values=[p.headers];
  sh.getRange(`A5:${last}5`).format={fill:'#192840',font:{name:'Malgun Gothic',size:10,color:'#FFFFFF',bold:true},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:32,wrapText:true};
  sh.getRange(`A6:${last}${end}`).values=p.rows;
  sh.getRange(`A6:${last}${end}`).setNumberFormat('#,##0');
  for(let c=0;c<p.headers.length;c++){
    sh.getRange(`${col(c)}1:${col(c)}${end}`).format.columnWidthPx=p.widths[c];
    const numeric=p.rows.some(row=>typeof row[c]==='number');
    sh.getRange(`${col(c)}6:${col(c)}${end}`).format.horizontalAlignment=numeric?'right':'left';
  }
  for(const [c,fmt]of Object.entries(p.formats))sh.getRange(`${c}6:${c}${end}`).setNumberFormat(fmt);
  for(let r=6;r<=end;r+=2)sh.getRange(`A${r}:${last}${r}`).format.fill='#F1F5FA';
  if(end>25)sh.freezePanes.freezeRows(5);
  sh.tables.add(`A5:${last}${end}`,true,'Table'+payload.sheets.indexOf(p));
  if(p.name==='유효성'){
    for(let r=6;r<=end;r++){
      for(const [c,formula]of Object.entries({E:`=C${r}/B${r}`,F:`=D${r}/C${r}`,G:`=D${r}/B${r}`,I:`=H${r}/${payload.scenario_count}`}))sh.getRange(`${c}${r}`).formulas=[[formula]];
    }
    sh.getRange(`E6:I${end}`).format.horizontalAlignment='right';
    sh.getRange('A14').values=[['유효 후보 없음은 환경 자체의 경로 부재를 증명하지 않습니다.']];
    sh.getRange('A15').values=[['연속 충돌 회피·수치 정확도의 보증과 이산 표본의 유효성은 구분합니다.']];
    const labels={has_valid:'유효 후보 있음',all_unreachable:'모두 도달 실패',all_reachable_colliding:'모두 충돌',mixed_unreachable_collision:'도달·충돌 혼합',other_calculation_or_schema:'기타 문제'};
    sh.getRange('A18:C18').values=[['상황 분류','상황 수','승자 집계 포함']];
    sh.getRange('A18:C18').format={fill:'#192840',font:{name:'Malgun Gothic',size:10,color:'#FFFFFF',bold:true},rowHeight:30,horizontalAlignment:'center',verticalAlignment:'center'};
    const failureRows=payload.no_valid_counts.map(r=>[labels[r.reason],r.scenarios,r.reason==='has_valid'?'포함':'제외']);
    sh.getRange('A19:C23').values=failureRows;
    sh.getRange('A19:C23').format={font:{name:'Malgun Gothic',size:10,color:'#20314D'},rowHeight:25};
    sh.getRange('B19:B23').setNumberFormat('#,##0');sh.getRange('B19:B23').format.horizontalAlignment='right';
    sh.getRange('C19:C23').format.horizontalAlignment='center';
  }
}
wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},maxChars:4000});
await fs.writeFile(path.join(previews,'formula_scan.json'),JSON.stringify(errors,null,2));
await fs.writeFile(path.join(previews,'formula_scan.ndjson'),errors.ndjson);
const inspected=await wb.inspect({kind:'table',range:'유효성!A5:I11',include:'values,formulas',tableMaxRows:8,tableMaxCols:9,maxChars:6000});
await fs.writeFile(path.join(previews,'summary_values.json'),JSON.stringify(inspected,null,2));
// Verify live ratio formulas react to a source count, then restore before export.
const summary=wb.worksheets.getItem('유효성');
const before=summary.getRange('E6').values[0][0];
const count=summary.getRange('C6').values[0][0];
summary.getRange('C6').values=[[count/2]];wb.recalculate();
const after=summary.getRange('E6').values[0][0];
if(Math.abs(after-before/2)>1e-12)throw new Error('Formula input perturbation failed');
summary.getRange('C6').values=[[count]];wb.recalculate();
await fs.writeFile(path.join(previews,'formula_recalculation_check.json'),JSON.stringify({before,after,restored:summary.getRange('E6').values[0][0],passed:true}));
for(const p of payload.sheets){
  const last=col(p.headers.length-1),end=5+p.rows.length;
  for(const [name,range]of [[p.name,`A1:${last}${p.name==='유효성'?24:Math.min(Math.max(end,16),29)}`],...(end>30?[[p.name+'_tail',`A${Math.max(6,end-12)}:${last}${end}`]]:[])]){
    const blob=await wb.render({sheetName:p.name,range,scale:1.5,format:'png'});
    await fs.writeFile(path.join(previews,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
    console.log('Rendered',name);
  }
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,'phase4_summary.xlsx'));
await fs.writeFile(path.join(out,'excel_build_environment.json'),JSON.stringify({node:process.version,library:'@oai/artifact-tool',runtimeBundle:'26.905.11957',sheets:payload.sheets.map(s=>s.name),source:'workbook_payload.json',nativeExcelApplicationTested:false},null,2));
console.log('Exported phase4_summary.xlsx');
