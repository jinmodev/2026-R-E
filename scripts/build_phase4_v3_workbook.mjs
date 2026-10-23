// Typed tables from the train-only Python report; bundled artifact-tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'outputs','phase4_v3_analysis');
const payload=JSON.parse(await fs.readFile(path.join(out,'workbook_payload.json'),'utf8'));
const wb=Workbook.create(),preview=path.join(out,'excel_previews');
await fs.mkdir(preview,{recursive:true});
const col=n=>String.fromCharCode(65+n);
let ratioChecks=0;
for(const p of payload.sheets){
  const sh=wb.worksheets.add(p.name),last=col(p.headers.length-1),end=p.rows.length+5;
  sh.showGridLines=false;if(p.name==='유효성')sh.tabColor='#192840';
  sh.getRange(`A1:${last}${end}`).format={font:{name:'Malgun Gothic',size:10,color:'#20314D'},rowHeight:25,verticalAlignment:'center'};
  sh.getRange('A2').values=[[p.title]];
  sh.getRange('A2').format.font={name:'Malgun Gothic',size:14,bold:true,color:'#192840'};
  sh.getRange(`A2:${last}2`).format.borders={bottom:{style:'thin',color:'#9CABBF'}};
  sh.getRange('A3').values=[[p.note]];
  sh.getRange('A3').format.font={name:'Malgun Gothic',size:10,color:'#53627B'};
  sh.getRange(`A5:${last}5`).values=[p.headers];
  sh.getRange(`A5:${last}5`).format={fill:'#192840',font:{name:'Malgun Gothic',size:10,color:'#FFFFFF',bold:true},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:34,wrapText:true};
  sh.getRange(`A6:${last}${end}`).values=p.rows;
  sh.getRange(`A6:${last}${end}`).setNumberFormat('#,##0');
  for(let c=0;c<p.headers.length;c++){
    sh.getRange(`${col(c)}1:${col(c)}${end}`).format.columnWidthPx=p.widths[c];
    sh.getRange(`${col(c)}6:${col(c)}${end}`).format.horizontalAlignment=p.rows.some(r=>typeof r[c]==='number')?'right':'left';
  }
  for(const [c,fmt] of Object.entries(p.formats))sh.getRange(`${c}6:${c}${end}`).setNumberFormat(fmt);
  for(let r=6;r<=end;r+=2)sh.getRange(`A${r}:${last}${r}`).format.fill='#F1F5FA';
  for(const [c,[num,den]] of Object.entries(p.ratios)){
    sh.getRange(`${c}6:${c}${end}`).format.horizontalAlignment='right';
    for(let r=6;r<=end;r++){
      sh.getRange(`${c}${r}`).formulas=[[`=IF(${den}${r}=0,"n.a.",${num}${r}/${den}${r})`]];
      const value=sh.getRange(`${c}${r}`).values[0][0];
      const n=p.rows[r-6][num.charCodeAt(0)-65],d=p.rows[r-6][den.charCodeAt(0)-65];
      if(d===0?value!=='n.a.':Math.abs(value-n/d)>1e-12)throw new Error('Ratio mismatch: '+p.name+c+r);
      ratioChecks++;
    }
  }
  if(end>25)sh.freezePanes.freezeRows(5);
  sh.tables.add(`A5:${last}${end}`,true,'V3Table'+payload.sheets.indexOf(p));
  if(p.name==='유효성'){
    sh.getRange('A14').values=[['¹ 모두 100%인 것은 검사 상태 정의 때문이며 안전 성능 비교가 아닙니다. 별도 격자진단 시트를 참조하세요.']];
    sh.getRange('A15').values=[['직선 2개, 나머지 종류 각 8개 후보. 후보 수 차이가 탐색 기회에 영향을 줍니다.']];
    sh.getRange('A16').values=[['검사 통과는 수학적 정확도·물리적 안전 인증이 아닙니다. 모든 최적은 준비된 42개 후보 안의 결과입니다.']];
    sh.getRange('A14:K16').format.font={name:'Malgun Gothic',size:10,color:'#53627B'};
  }
  if(p.name==='격자진단'){
    sh.getRange('A14').values=[['¹ 두 격자 모두 도달 가능한 후보 중 충돌 표본이 없는 후보. 미검사 표본을 안전으로 세지 않습니다.']];
    sh.getRange('A14:G14').format.font={name:'Malgun Gothic',size:10,color:'#53627B'};
  }
}
const sh=wb.worksheets.getItem('유효성');
const original=sh.getRange('C6').values[0][0],before=sh.getRange('F6').values[0][0];
sh.getRange('C6').values=[[original/2]];wb.recalculate();
const after=sh.getRange('F6').values[0][0];if(Math.abs(after-before/2)>1e-12)throw new Error('Input change did not recalculate');
sh.getRange('C6').values=[[0]];wb.recalculate();
if(sh.getRange('G6').values[0][0]!=='n.a.')throw new Error('Zero denominator not exposed');
sh.getRange('C6').values=[[original]];wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},maxChars:3000});
await fs.writeFile(path.join(preview,'formula_scan.json'),JSON.stringify(errors,null,2));
const summary=await wb.inspect({kind:'table',range:'유효성!A5:K11',include:'values,formulas',tableMaxRows:7,tableMaxCols:11,maxChars:7000});
await fs.writeFile(path.join(preview,'summary_values.json'),JSON.stringify(summary,null,2));
await fs.writeFile(path.join(preview,'recalculation.json'),JSON.stringify({ratioChecks,before,after,restored:sh.getRange('F6').values[0][0],zeroDenominator:'n.a.',passed:true}));
for(const p of payload.sheets){
  const last=col(p.headers.length-1),end=p.rows.length+5;
  for(const [suffix,range] of [['',`A1:${last}${Math.min(Math.max(end,16),22)}`],...(end>22?[['_tail',`A${end-10}:${last}${end}`]]:[])]){
    const blob=await wb.render({sheetName:p.name,range,scale:1.5,format:'png'});
    await fs.writeFile(path.join(preview,p.name+suffix+'.png'),new Uint8Array(await blob.arrayBuffer()));
    console.log('Rendered',p.name+suffix);
  }
}
wb.recalculate();
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,'phase4_v3_summary.xlsx'));
await fs.writeFile(path.join(out,'excel_build_environment.json'),JSON.stringify({node:process.version,library:'@oai/artifact-tool',runtimeBundle:'26.909.12148',ratioChecks,nativeExcelApplicationTested:false},null,2));
console.log('Workbook exported');
