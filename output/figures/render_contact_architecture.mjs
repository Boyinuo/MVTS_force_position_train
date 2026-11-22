import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const out=dirname(fileURLToPath(import.meta.url));
const browser='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const result=spawnSync(browser,[
  '--headless','--disable-gpu','--no-sandbox','--no-first-run','--no-default-browser-check',
  '--disable-extensions','--disable-background-networking','--disable-component-update',
  '--force-device-scale-factor=1','--hide-scrollbars','--window-size=3200,1560',
  '--virtual-time-budget=2000','--dump-dom',
  `--user-data-dir=${join(out,'.svg-render-profile')}`,
  `--screenshot=${join(out,'contact_aware_architecture_svg_preview.png')}`,
  pathToFileURL(join(out,'contact_aware_architecture_qa.html')).href
],{encoding:'utf8',timeout:25000,maxBuffer:4*1024*1024,windowsHide:true});
if(result.error) throw result.error;
const match=result.stdout.match(/<pre id="svg-qa-report">([\s\S]*?)<\/pre>/);
if(!match) {
  console.error(result.stderr.slice(-1800));
  throw new Error('Browser did not return SVG layout measurements');
}
const report=JSON.parse(match[1].replaceAll('&amp;','&').replaceAll('&lt;','<').replaceAll('&gt;','>'));
writeFileSync(join(out,'contact_aware_architecture_svg_qa.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
const svg=readFileSync(join(out,'contact_aware_architecture.svg'),'utf8');
if(/<image\b|data:image|<script\b/.test(svg))throw new Error('SVG contains a raster image or script');
if(report.outside.length||report.overlaps.length||report.stroke_hits.length)process.exitCode=1;
