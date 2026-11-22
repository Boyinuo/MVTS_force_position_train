/** Editable vector reconstruction of the proposed contact-aware estimator.
 * All waveforms and sensor thumbnails are illustrative, not measured data.
 * Run: node output/figures/draw_contact_architecture.mjs
 */
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const out = dirname(fileURLToPath(import.meta.url));
const W = 3200, H = 1560;
const p = [];
const palette = { ink:'#2d2d39', muted:'#647183', border:'#bec6d6',
  mint:'#e6f3ec', mintLine:'#85b69c', green:'#487c63', blue:'#e4f1fb',
  blueLine:'#89bad8', navy:'#386f91', purple:'#eee5f5', rose:'#fae8ed',
  orange:'#fff0d9', amber:'#b87917', teal:'#007d82', tealFill:'#e3f4f3' };
const esc = s => String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
let serial = 0;
function group(id, fn) { p.push(`<g id="${id}">`); fn(); p.push('</g>'); }
function rect(x,y,w,h,fill='#fff',stroke=palette.ink,r=16,extra='') {
  p.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="2" ${extra}/>`);
}
function text(x,y,s,size=26,weight='normal',anchor='middle',fill=palette.ink,extra='') {
  p.push(`<text id="label-${++serial}" x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" fill="${fill}" ${extra}>${esc(s)}</text>`);
}
function lines(x,y,arr,size=26,step=34,weight='normal',anchor='middle',fill=palette.ink) {
  arr.forEach((s,i)=>text(x,y+i*step,s,size,weight,anchor,fill));
}
function path(d,color=palette.ink,width=2.8,arrow=true,dash='') {
  const marker = color===palette.amber?'amber':color===palette.teal?'teal':'ink';
  p.push(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round" ${arrow?`marker-end="url(#arrow-${marker})"`:''} ${dash?`stroke-dasharray="${dash}"`:''}/>`);
}
function line(x1,y1,x2,y2,color=palette.ink,width=2.8,arrow=true,dash='') {
  path(`M ${x1} ${y1} L ${x2} ${y2}`,color,width,arrow,dash);
}
function pill(x,y,w,label,fill='#fcfdff',size=25) {
  rect(x,y,w,48,fill,palette.border,11); text(x+w/2,y+33,label,size);
}
function panel(id,x,y,w,h,title,dashed=false) {
  group(id,()=>{rect(x,y,w,h,'#fff',palette.border,20,dashed?'stroke-dasharray="9 6"':'');
    text(x+20,y+43,title,32,'bold','start');});
}
function card(id,x,y,w,h,title,body,fill,size=27) {
  group(id,()=>{rect(x,y,w,h,fill); text(x+w/2,y+40,title,size,'bold');
    lines(x+w/2,y+82,body,25,34);});
}
function sensor(id,x,y,w,h) {
  group(id,()=>{
    rect(x,y,w,h,'#e5f0ec',palette.mintLine,16);
    rect(x+7,y+7,w-14,h-14,'url(#sensor-field)','#4e9199',12);
    p.push(`<ellipse cx="${x+w*.53}" cy="${y+h*.57}" rx="${w*.31}" ry="${h*.24}" fill="url(#contact-glow)"/>`);
    for(let j=0;j<7;j++) {
      let d=''; for(let i=0;i<=30;i++){const xx=x+15+(w-30)*i/30; const yy=y+22+j*(h-44)/6-5*Math.exp(-(((i-16)/7)**2));d+=`${i?'L':'M'} ${xx.toFixed(2)} ${yy.toFixed(2)} `;}
      path(d,'#42a1be',1.4,false);
      for(let i=0;i<5;i++) p.push(`<circle cx="${x+18+i*(w-36)/4}" cy="${y+22+j*(h-44)/6}" r="2.5" fill="#176277"/>`);
    }
  });
}
function waves(x1,x2,y,gap,count,amplitude=5,stroke='#335a6d') {
  for(let j=0;j<count;j++) {
    let d=''; for(let i=0;i<=150;i++) {
      const u=i/150, xx=x1+u*(x2-x1);
      const envelope=.25+.75*Math.exp(-(((u-.67)/.18)**2));
      const yy=y+j*gap+amplitude*envelope*(Math.sin(i*.67+j*.5)+.35*Math.sin(i*1.73+j));
      d+=`${i?'L':'M'} ${xx.toFixed(2)} ${yy.toFixed(2)} `;
    } path(d,stroke,1.9,false);
  }
}

p.push(`<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="figure-title figure-description">
<title id="figure-title">Contact-Aware Dual-Rate Visual–Acoustic State Estimation</title>
<desc id="figure-description">Editable vector schematic. Causal visual and acoustic encoders feed joint cross-attention and a stateful GRU. Four learned heads predict contact, position, wrench and slip. A separate delayed contact-state cache supports deterministic transition logic and physical-unit output gating. Ground-truth contact labels are used only in training losses. Waveforms and tactile icons are illustrative.</desc>
<metadata>Proposed architecture, vector reconstruction, 2026-11-21. No measured data or performance results are depicted.</metadata>
<defs>
<marker id="arrow-ink" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2d2d39"/></marker>
<marker id="arrow-teal" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#007d82"/></marker>
<marker id="arrow-amber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#b87917"/></marker>
<linearGradient id="sensor-field" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#0f7887"/><stop offset=".6" stop-color="#237e9d"/><stop offset="1" stop-color="#5c4487"/></linearGradient>
<radialGradient id="contact-glow"><stop offset="0" stop-color="#efdf42" stop-opacity=".95"/><stop offset=".45" stop-color="#c4d73f" stop-opacity=".65"/><stop offset="1" stop-color="#7bc36b" stop-opacity="0"/></radialGradient>
</defs>
<style>text{font-family:"Times New Roman","Cambria","DejaVu Serif",serif} .small{font-size:22px} path,rect,line,ellipse,circle{vector-effect:non-scaling-stroke}</style>`);
rect(0,0,W,H,'#fff','none',0);
text(1600,65,'Contact-Aware Dual-Rate Visual–Acoustic State Estimation',47,'bold');
line(35,53,520,53,palette.ink,1.6,false); line(2680,53,3165,53,palette.ink,1.6,false);

panel('panel-a',25,120,480,1060,'(a) Causal dual-rate inputs');
panel('panel-b',525,120,1260,1060,'(b) Shared encoding and temporal fusion');
panel('panel-c',1805,120,1370,1060,'(c) Contact prediction and rule-based output gating');
panel('panel-d',25,1205,3150,300,'(d) Training supervision · ground truth is not an inference input',true);

group('causal-timing',()=>{
  rect(45,190,440,310,'#fafdfc',palette.border,14);
  text(265,229,'Camera frames · 15 Hz',30,'bold');
  sensor('camera-previous',95,263,95,133); sensor('camera-current',345,263,95,133);
  text(265,326,'···',42);text(142,434,'Iⱼ₋₁',29);text(392,434,'Iⱼ',29);
  text(265,477,'Reuse the latest available image',25,'normal');
  text(265,530,'Six-channel audio · 50 kHz',30,'bold');
  // Equal 80-ms windows shifted by 6.67 ms: ~91.7% overlap.
  rect(75,650,325,180,'#eee8f7','#9674b7',4,'fill-opacity=".56"');
  rect(102,650,325,180,'#d9edfb','#498fc0',4,'fill-opacity=".52"');
  waves(60,427,669,29,6,5);
  path('M 75 596 V 584 H 400 V 596','#9674b7',2,false);
  text(237,575,'Aₜ₋₁',28,'normal','middle','#81619e');
  path('M 102 642 V 630 H 427 V 642','#498fc0',2,false);
  text(265,618,'Aₜ',28,'normal','middle','#2e79aa');
  line(400,650,400,865,'#9674b7',1.8,false,'6 5');
  line(427,650,427,899,'#498fc0',1.8,false,'6 5');
  line(55,846,475,846,palette.ink,2.5);
  text(396,884,'tₜ₋₁',27,'normal','end');text(432,918,'tₜ',27);
  text(474,878,'Time',23,'normal','end');
  pill(45,953,440,'4000 samples/channel · 80 ms look-back',palette.blue,24);
  pill(45,1014,440,'Nominal 150 Hz updates · 6.67 ms',palette.blue,24);
  lines(265,1104,['Host camera timestamps;','exposure timing unavailable'],24,30,'normal','middle',palette.muted);
  text(265,1170,'Waveforms are schematic',22,'normal','middle',palette.muted);
});

// Pathway backgrounds; learned data-flow connections are routed before cards.
rect(550,190,1210,310,'#f3faf6','#afcebd',18);
text(575,231,'Visual path: 15 Hz · encode each new image once',29,'bold','start',palette.green);
rect(550,580,710,365,'#f4faff','#adcfe5',18);
text(570,622,'Acoustic path: nominal 150 Hz',29,'bold','start',palette.navy);

group('learned-flow',()=>{
  line(760,355,790,355);line(995,355,1030,355);
  path('M 1280 355 H 1405 V 580');
  line(700,770,730,770);line(860,770,890,770);line(1075,770,1105,770);
  line(1245,770,1290,770);line(1520,770,1560,770);
  path('M 1755 770 H 1835');
  path('M 1835 282 V 1080',palette.ink,2.8,false);
  [282,650,865,1080].forEach(y=>line(1835,y,1875,y));
  // Only h_t feeds the delayed hidden memory; fusion features do not.
  path('M 1770 770 V 1063 H 1755');
  path('M 1622 1020 V 945');
  line(2110,282,2350,282);
  line(2800,282,2860,282);
  [[650,610],[865,825],[1080,1040]].forEach(([cy,ry])=>{
    line(2110,cy,2160,cy);line(2300,cy,2340,cy);line(2815,cy,2870,cy);
  });
});

group('visual-encoder',()=>{
  rect(575,255,185,220,palette.mint);
  text(667.5,289,'Latest image',27,'bold');sensor('latest-tactile-icon',620,303,95,108);
  pill(592,419,151,'3 × H × W','#fcfffd',25);
  p.push('<path d="M 790 255 L 995 283 L 995 447 L 790 475 Z" fill="#e3f2e8" stroke="#2d2d39" stroke-width="2"/>');
  lines(892,330,['Visual','ResNet-18'],31,40,'bold');
  text(892,416,'4 × 4 pool + Linear',25);
  rect(1030,255,250,220,palette.mint);
  text(1155,291,'Cached visual tokens',27,'bold');
  for(let j=0;j<4;j++)for(let i=0;i<4;i++)rect(1086+i*37,310+j*25,27,19,'#b9dccb','#80a991',5);
  pill(1050,420,210,'Vⱼ: 16 × 128','#fcfffd',26);
  text(1540,300,'Visual context',29,'bold','middle',palette.green);
  text(1540,345,'Position / deformation',26,'normal','middle',palette.green);
  lines(1515,415,['Cache tokens until','a new image arrives'],25,34,'normal','middle',palette.muted);
});

group('acoustic-encoder',()=>{
  rect(570,650,130,240,palette.blue);lines(635,687,['6-ch','causal audio'],25,32,'bold');
  waves(589,681,760,19,4,3);pill(578,835,114,'6 × 4000','#fcfdff',23);
  rect(730,650,130,240,palette.orange);lines(795,687,['STFT +','Log-Mel'],25,32,'bold');
  const heat=['#4f2466','#78317e','#b65078','#e57964','#f2a45c','#ffc65c'];
  for(let j=0;j<7;j++)for(let i=0;i<6;i++)rect(754+i*14,750+j*10,14,10,heat[(i+2*j)%heat.length],'none',0);
  pill(738,835,114,'6 × 64 × 28','#fffdfa',22);
  rect(890,650,185,240,palette.blue);
  lines(982.5,686,['Shared','per-mic','ResNet-18'],27,33,'bold');
  lines(982.5,797,['Frequency-only','strides'],24,28);pill(901,835,163,'6 × 28 × 128','#fcfdff',23);
  rect(1105,650,140,240,palette.blue);lines(1175,693,['Mean over','microphones'],22,33,'bold');
  text(1175,791,'Weights: 1/6',23);pill(1111,835,128,'Sₜ: 28 × 128','#fcfdff',22);
});

group('joint-attention',()=>{
  rect(1290,580,230,365,palette.purple);
  lines(1405,624,['Joint','cross-attention'],30,38,'bold');
  lines(1405,720,['Position + modality','embeddings'],23,29);
  lines(1405,796,['Q = Linear(last','audio token)'],23,30);
  text(1405,860,'K,V = [Sₜ; Vⱼ]: 44 × 128',21);
  pill(1304,881,202,'fₜ: 1 × 128','#fcfaff',25);
});
group('stateful-gru',()=>{
  rect(1560,580,195,365,palette.purple);
  lines(1657.5,657,['Stateful','GRU(128,256)'],29,43,'bold');
  lines(1657.5,793,['Temporal','state memory'],26,36);
  pill(1574,881,167,'hₜ: 256-D','#fcfaff',26);
  text(1798,744,'hₜ',29);
  rect(1490,1020,265,85,'#f7f1fb');text(1622.5,1073,'GRU memory hₜ₋₁',27,'bold');
  lines(1458,1003,['hₜ stored for next step'],23,30,'normal','end',palette.muted);
  text(1155,1150,'Carry h across continuous steps; no reset at contact transitions',27,'normal','middle',palette.muted);
});

card('contact-head',1875,205,235,155,'Contact head',['MLP 256 → 1','+ Sigmoid'],palette.purple,29);
text(2228,259,'pₜ = P(contact)',26);
group('contact-transition',()=>{
  rect(2350,205,450,155,palette.tealFill,palette.teal,18);
  text(2575,246,'Hysteresis + transition logic',29,'bold');
  text(2575,293,'θoff < θon',29);text(2575,335,'(ĉₜ₋₁, ĉₜ)',29);
  rect(2250,405,370,95,palette.tealFill,palette.teal,16);
  text(2435,442,'Contact state cache',29,'bold');text(2435,481,'ĉₜ → ĉₜ₋₁ at next step',27);
  path('M 2400 360 V 405',palette.teal,2.8);
  path('M 2620 452 H 2690 V 360',palette.teal,2.8);
  path('M 2790 360 V 560',palette.amber,3,true,'8 7');
  rect(2860,222,275,120,'#fff');
  lines(2997.5,260,['Contact probability','pₜ + state ĉₜ'],27,42);
});

card('position-head',1875,575,235,150,'Position head',['MLP 256 → 3','(xₜ, yₜ, zₜ)'],palette.purple,29);
group('wrench-head',()=>{
  rect(1875,790,235,150,palette.rose);
  text(1992.5,830,'Wrench head',29,'bold');text(1992.5,872,'MLP 256 → 6',25);
  text(1992.5,903,'(Fx, Fy, Fz,',24);text(1992.5,930,'Mx, My, Mz)',24);
});
card('slip-head',1875,1005,235,150,'Slip head',['MLP 256 → 2','(Δsₓ, Δsᵧ)'],palette.orange,29);
// Keep the long six-axis list within the wrench card using a separate small run.
// All labels stay editable text; no outlines or embedded raster imagery.
group('physical-unit-conversion',()=>{
  for(const y of [610,825,1040]) {
    rect(2160,y,140,80,'#fff',palette.ink,12);
    lines(2230,y+30,['De-','normalize'],24,29);
  }
});

group('physical-output-gate',()=>{
  rect(2340,560,475,595,'#fff9ef',palette.amber,18);
  text(2577.5,602,'Physical output gate',31,'bold');
  text(2577.5,638,'Controlled by contact transition',24);
  rect(2358,665,439,410,'#fff',palette.ink,9);
  line(2438,665,2438,1075,palette.ink,1.3,false);
  [710,795,895,985].forEach(y=>line(2358,y,2797,y,palette.ink,1.3,false));
  text(2398,695,'ĉₜ₋₁→ĉₜ',23);text(2617.5,695,'Gate decision',25,'bold');
  text(2398,757,'0 → 0',28);
  lines(2453,743,['No contact: wrench = 0;','position / slip invalid'],25,31,'normal','start');
  text(2398,852,'0 → 1',28);
  lines(2453,824,['Contact onset: enable','regressions; initialize','slip reference'],25,30,'normal','start');
  text(2398,950,'1 → 1',28);
  lines(2453,924,['Sustained contact: normal','position / wrench / slip'],25,31,'normal','start');
  text(2398,1038,'1 → 0',28);
  lines(2453,1015,['Contact release: wrench = 0;','position / slip invalid'],25,31,'normal','start');
  lines(2577.5,1112,['Slip increment invalid across onset','or missing history'],24,29);
});
group('regression-outputs',()=>{
  for(const [id,y,labels] of [
    ['contact-position-output',610,['Contact position','+ valid']],
    ['contact-wrench-output',825,['Contact wrench']],
    ['slip-output',1040,['Slip increment','+ valid']]]) {
    rect(2870,y,270,90,'#fff');
    lines(3005,y+(labels.length===1?54:35),labels,28,35);
  }
});

group('legend',()=>{
  text(1865,545,'Legend:',23,'bold','start');line(1960,537,2020,537,palette.ink,2.7);
  text(2040,545,'Learned flow',23,'normal','start');
  line(2215,537,2275,537,palette.amber,2.7,true,'7 5');text(2295,545,'Rule control',23,'normal','start');
  line(2485,537,2545,537,palette.teal,2.7);text(2565,545,'Delayed contact state',23,'normal','start');
});

group('training-only-supervision',()=>{
  rect(55,1285,670,175,'#fbf9fd',palette.border,16);
  text(390,1325,'Existing data labels',29,'bold');
  text(390,1370,'Press backgrounds: c*ₜ = 0',28);
  text(390,1415,'Verified press targets + continuous slides: c*ₜ = 1',27);
  rect(800,1310,520,120,palette.purple,palette.border,16);
  text(1060,1350,'Contact loss',29,'bold');text(1060,1400,'Lcontact = BCE(pₜ, c*ₜ)',29);
  text(1060,1292,'pₜ',27);line(1060,1298,1060,1310,'#77618f',2,false,'5 4');
  rect(1380,1310,700,120,palette.purple,palette.border,16);
  text(1730,1350,'No-contact wrench loss',29,'bold');
  text(1730,1400,'Lzero = mean[c*ₜ = 0] Huber(w̃ₜ / s)',29);
  text(1730,1292,'w̃ₜ (raw, de-normalized)',27);
  line(1730,1298,1730,1310,'#77618f',2,false,'5 4');
  rect(2150,1285,990,175,'#fbf9fd',palette.border,16);
  text(2645,1325,'Regression losses',29,'bold');
  lines(2190,1366,['Position: valid contact labels   ·   Wrench: calibrated targets',
    'Slip: valid within-contact increments',
    'w̃ₜ: physical-unit wrench BEFORE gating; per-axis scale s'],27,36,'normal','start');
  text(1590,1490,'Contact truth is used for supervision only; inference uses predicted states and a deterministic output gate.',26,'normal','middle',palette.muted);
});
text(3170,1540,'Four learned heads · deterministic contact transitions · physical-unit gating',26,'normal','end',palette.muted);
p.push('</svg>');
const svg = p.join('\n');
writeFileSync(join(out,'contact_aware_architecture.svg'),svg,'utf8');

// Local browser QA: inspect vector text bounds and pairs, without changing SVG.
const qa = `<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;background:white}svg{display:block}</style></head><body>${svg.replace(/^<\?xml[^>]+>\s*/,'')}<script>
document.fonts.ready.then(()=>{
 const nodes=[...document.querySelectorAll('svg text')];
 const records=nodes.map(n=>{const b=n.getBBox();return {id:n.id,text:n.textContent,x:b.x,y:b.y,w:b.width,h:b.height}});
 const outside=records.filter(b=>b.x<0||b.y<0||b.x+b.w>${W}||b.y+b.h>${H});
 const overlaps=[];for(let i=0;i<records.length;i++)for(let j=i+1;j<records.length;j++){
  const a=records[i],b=records[j],dx=Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x),dy=Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y);
  if(dx>2&&dy>2)overlaps.push({a:a.text,b:b.text,dx,dy});
 }
 const stroke_hits=[];
 for(const route of document.querySelectorAll('svg path[fill="none"]')){
  const length=route.getTotalLength();const hit=new Set();
  for(let s=0;s<=length;s+=2){const q=route.getPointAtLength(s);
   for(const b of records){if(q.x>b.x+2&&q.x<b.x+b.w-2&&q.y>b.y+2&&q.y<b.y+b.h-2)hit.add(b.text);}
  }
  if(hit.size)stroke_hits.push({path:route.getAttribute('d').slice(0,100),text:[...hit]});
 }
 const report={width:${W},height:${H},text_count:records.length,raster_images:document.querySelectorAll('svg image').length,outside,overlaps,stroke_hits};
 const pre=document.createElement('pre');pre.id='svg-qa-report';pre.textContent=JSON.stringify(report);document.body.append(pre);
});</script></body></html>`;
writeFileSync(join(out,'contact_aware_architecture_qa.html'),qa,'utf8');
console.log(JSON.stringify({svg:join(out,'contact_aware_architecture.svg'),elements:serial,width:W,height:H}));
