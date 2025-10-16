const $ = s=>document.querySelector(s);
/* elements */
const pdfInput=$('#pdf-file'), btnLoadPdf=$('#btn-load-pdf');
const imgInput=$('#img-files'), btnLoadImgs=$('#btn-load-imgs');
const loadStatus=$('#load-status');
const workArea=$('#work-area'), extractArea=$('#extract-area');
const canvas=$('#page-canvas'), ctx=canvas.getContext('2d',{willReadFrequently:true});
const regionSel=$('#region-name'); const btnRect=$('#mode-rect'), btnTop=$('#mode-top'), btnBot=$('#mode-bot');
const applyAll=$('#apply-all'); const btnClear=$('#btn-clear');
const prevBtn=$('#prev-page'), nextBtn=$('#next-page'), pageInfo=$('#page-info');
const zoomOut=$('#zoom-out'), zoomIn=$('#zoom-in'), zoom100=$('#zoom-100'), zoomLabel=$('#zoom-label');
const topDepthEl=$('#top-depth'), botDepthEl=$('#bot-depth');
const snapTopEl=$('#snap-top'), snapBotEl=$('#snap-bot'), clipDepthEl=$('#clip-depth');
const sepFracEl=$('#sep-frac'), minBandEl=$('#min-band'), btnPreview=$('#btn-preview');
const btnSave=$('#btn-save'), btnLoadCfg=$('#btn-loadcfg');
const btnExtract=$('#btn-extract'), extractStatus=$('#extract-status');

/* state */
const ST={source:'none', pdfDoc:null, pages:[], pdfPages:[], pdfVP:[], curr:0, zoom:1, renderScale:5.0,
  regions:{}, depthPx:{}, topFt:0, botFt:36, snapTop:true, snapBot:true, clipDepth:true, sepFrac:0.45, minBand:40, pageOffsets:{}};
function regs(i){ if(!ST.regions[i]) ST.regions[i]={}; return ST.regions[i]; }
function dpx(i){ if(!ST.depthPx[i]) ST.depthPx[i]={top_y:null,bot_y:null}; return ST.depthPx[i]; }
function setMode(m){ ST.mode=m; [btnRect,btnTop,btnBot].forEach(b=>b.classList.remove('active')); if(m==='rect')btnRect.classList.add('active'); if(m==='top')btnTop.classList.add('active'); if(m==='bot')btnBot.classList.add('active'); }
btnRect.onclick=()=>setMode('rect'); btnTop.onclick=()=>setMode('top'); btnBot.onclick=()=>setMode('bot');

/* zoom */
function updateZoom(){ zoomLabel.textContent=Math.round(ST.zoom*100)+'%'; draw(); }
function fitWidth(){ const P=ST.pages[ST.curr]; if(!P) return; const W=document.querySelector('.canvas-wrap').clientWidth||P.w; ST.zoom=Math.min(1, W/P.w); updateZoom(); }
zoomOut.onclick=()=>{ ST.zoom=Math.max(.2, ST.zoom/1.2); updateZoom(); };
zoomIn.onclick=()=>{ ST.zoom=Math.min(6, ST.zoom*1.2); updateZoom(); };
zoom100.onclick=()=>fitWidth();

/* draw */
function draw(){
  const P=ST.pages[ST.curr]; if(!P) return; canvas.width=Math.round(P.w*ST.zoom); canvas.height=Math.round(P.h*ST.zoom);
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(P.canvas,0,0,canvas.width,canvas.height);
  const r=regs(ST.curr); Object.entries(r).forEach(([name,rel])=>{ const [x0,y0,x1,y1]=relToAbs(rel,P.w,P.h); strokeRect(name,x0,y0,x1,y1); });
  const dp=dpx(ST.curr); if(dp.top_y!=null) strokeH(dp.top_y,'#00FFFF'); if(dp.bot_y!=null) strokeH(dp.bot_y,'#FF00FF');
  pageInfo.textContent=`${ST.curr+1} / ${ST.pages.length}`;
}
function strokeRect(name,x0,y0,x1,y1){ const sx=x0*ST.zoom, sy=y0*ST.zoom, ex=x1*ST.zoom, ey=y1*ST.zoom;
  ctx.save(); ctx.lineWidth=3; ctx.strokeStyle=name==='description_col'?'#00C853':(name==='nvalue_col'?'#FF1744':'#FFD54F'); ctx.strokeRect(sx,sy,ex-sx,ey-sy); ctx.restore(); }
function strokeH(y,c){ const sy=y*ST.zoom; ctx.save(); ctx.strokeStyle=c; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(0,sy); ctx.lineTo(canvas.width,sy); ctx.stroke(); ctx.restore(); }
function relToAbs(rel,W,H){ const [a,b,c,d]=rel; return [a*W,b*H,c*W,d*H]; }
function absToRel(abs,W,H){ const [a,b,c,d]=abs; return [a/W,b/H,c/W,d/H]; }

/* mouse */
let drag=null;
canvas.addEventListener('mousedown',e=>{ if(ST.mode!=='rect') return; const r=canvas.getBoundingClientRect(); drag={x0:(e.clientX-r.left)/ST.zoom, y0:(e.clientY-r.top)/ST.zoom, x1:0,y1:0}; });
canvas.addEventListener('mousemove',e=>{ if(!drag) return; const r=canvas.getBoundingClientRect(); drag.x1=(e.clientX-r.left)/ST.zoom; drag.y1=(e.clientY-r.top)/ST.zoom; draw();
  const x=Math.min(drag.x0,drag.x1)*ST.zoom, y=Math.min(drag.y0,drag.y1)*ST.zoom, w=Math.abs(drag.x1-drag.x0)*ST.zoom, h=Math.abs(drag.y1-drag.y0)*ST.zoom;
  ctx.save(); ctx.setLineDash([6,4]); ctx.strokeStyle='#4db6ac'; ctx.strokeRect(x,y,w,h); ctx.restore(); });
canvas.addEventListener('mouseup',()=>{ if(!drag) return; const P=ST.pages[ST.curr]; const x0=Math.min(drag.x0,drag.x1), y0=Math.min(drag.y0,drag.y1), x1=Math.max(drag.x0,drag.x1), y1=Math.max(drag.y0,drag.y1);
  const rel=absToRel([x0,y0,x1,y1],P.w,P.h); if(applyAll.checked){ for(let i=0;i<ST.pages.length;i++) regs(i)[regionSel.value]=rel.slice(); } else { regs(ST.curr)[regionSel.value]=rel; } drag=null; draw(); });
canvas.addEventListener('click',e=>{ if(ST.mode!=='top' && ST.mode!=='bot') return; const r=canvas.getBoundingClientRect(); const y=(e.clientY-r.top)/ST.zoom; const dp=dpx(ST.curr);
  if(ST.mode==='top') dp.top_y=y; else dp.bot_y=y; if(applyAll.checked){ for(let i=0;i<ST.pages.length;i++){ const d=dpx(i); if(ST.mode==='top') d.top_y=y; else d.bot_y=y; } } draw(); });
btnClear.onclick=()=>{ delete regs(ST.curr)[regionSel.value]; if(applyAll.checked){ for(let i=0;i<ST.pages.length;i++) delete regs(i)[regionSel.value]; } draw(); };

/* PDF */
async function renderPdf(file){
  if(typeof pdfjsLib==='undefined'){ loadStatus.textContent='PDF engine blocked. Use a local server or Load Images.'; return; }
  const buf=await file.arrayBuffer(); const doc=await pdfjsLib.getDocument({data:buf, disableWorker:true}).promise;
  const n=doc.numPages; ST.pages=new Array(n); ST.pdfPages=new Array(n); ST.pdfVP=new Array(n);
  loadStatus.textContent=`Rendering ${n} page(s)…`;
  for(let i=1;i<=n;i++){ const page=await doc.getPage(i); const vp=page.getViewport({scale:ST.renderScale}); const cnv=document.createElement('canvas'); cnv.width=vp.width; cnv.height=vp.height;
    await page.render({canvasContext:cnv.getContext('2d',{willReadFrequently:true}), viewport:vp}).promise; ST.pages[i-1]={w:cnv.width,h:cnv.height,canvas:cnv}; ST.pdfPages[i-1]=page; ST.pdfVP[i-1]=vp; loadStatus.textContent=`Rendered ${i}/${n}`; }
  ST.pdfDoc=doc; ST.curr=0; ST.source='pdf'; workArea.classList.remove('hidden'); extractArea.classList.remove('hidden'); fitWidth();
}
btnLoadPdf.onclick=async ()=>{ const f=pdfInput.files?.[0]; if(!f){ loadStatus.textContent='Choose a PDF first.'; return; } try{ await renderPdf(f); loadStatus.textContent='Ready.'; }catch(e){ console.error(e); loadStatus.textContent='PDF load failed.'; } };

/* Images */
async function renderImages(files){ const list=[...files].sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true,sensitivity:'base'}));
  const imgs=await Promise.all(list.map(f=>new Promise((res,rej)=>{ const im=new Image(); im.onload=()=>res(im); im.onerror=rej; im.src=URL.createObjectURL(f);})));
  ST.pages=imgs.map(im=>{ const c=document.createElement('canvas'); c.width=im.naturalWidth; c.height=im.naturalHeight; c.getContext('2d').drawImage(im,0,0); return {w:c.width,h:c.height,canvas:c}; });
  ST.curr=0; ST.source='img'; workArea.classList.remove('hidden'); extractArea.classList.remove('hidden'); loadStatus.textContent=`Loaded ${ST.pages.length} image page(s).`; fitWidth();
}
btnLoadImgs.onclick=async ()=>{ const fs=imgInput.files; if(!fs||!fs.length){ loadStatus.textContent='Select PNG/JPG files.'; return; } try{ await renderImages(fs); }catch(e){ loadStatus.textContent='Image load failed.'; } };

/* nav */
prevBtn.onclick=()=>{ if(ST.curr>0){ ST.curr--; draw(); fitWidth(); } };
nextBtn.onclick=()=>{ if(ST.curr<ST.pages.length-1){ ST.curr++; draw(); fitWidth(); } };

/* OCR */
async function ocrText(c,psm=6){ const {data:{text}}=await Tesseract.recognize(c,'eng',{tessedit_pageseg_mode:psm}); return (text||'').replace(/\s+/g,' ').trim(); }
async function ocrDigitsWithPos(c){ const res=await Tesseract.recognize(c,'eng',{tessedit_pageseg_mode:6}); const out=[]; for(const w of (res.data.words||[])){ const s=(w.text||'').trim(); if(/^\d{1,3}$/.test(s)) out.push({val:parseInt(s,10), x:w.bbox.x0, y:w.bbox.y0, w:w.bbox.x1-w.bbox.x0, h:w.bbox.y1-w.bbox.y0}); } return out; }

/* bands */
function splitBandsWithParams(cnv, sepFrac, minBand){
  const w=cnv.width,h=cnv.height,g=cnv.getContext('2d').getImageData(0,0,w,h).data;
  const dark=new Uint8Array(w*h); for(let y=0;y<h;y++){ for(let x=0;x<w;x++){ const i=(y*w+x)<<2; const gray=0.299*g[i]+0.587*g[i+1]+0.114*g[i+2]; dark[y*w+x]=gray<190?1:0; } }
  const frac=new Float32Array(h), segs=new Uint16Array(h); for(let y=0;y<h;y++){ let cov=0,s=0,run=0; for(let x=0;x<w;x++){ const v=dark[y*w+x]; if(v){cov++;run++;} else { if(run>0){s++;run=0;} } } if(run>0)s++; frac[y]=cov/w; segs[y]=s; }
  const smooth=new Float32Array(h); for(let y=0;y<h;y++){ const a=frac[Math.max(0,y-1)],b=frac[y],c=frac[Math.min(h-1,y+1)]; smooth[y]=(a+b+c)/3; }
  const strong=Math.max(sepFrac,0.45), dashed=0.25, dashedSegs=Math.max(10,Math.floor(w/60)); const mask=new Uint8Array(h);
  for(let y=0;y<h;y++) mask[y]=(smooth[y]>=strong || (smooth[y]>=dashed && segs[y]>=dashedSegs))?1:0;
  const seps=[]; let s=-1; for(let y=0;y<h;y++){ if(mask[y]&&s<0)s=y; if((!mask[y]||y===h-1)&&s>=0){ const e=mask[y]?y:y-1; seps.push(Math.round((s+e)/2)); s=-1; } }
  const cuts=[0,...seps,h-1], bands=[]; for(let i=0;i<cuts.length-1;i++){ const a=cuts[i], b=cuts[i+1]; if(b-a>=Math.max(6,minBand)) bands.push([a,b]); } return {bands,seps};
}

/* utils */
function crop(i, rel){ const P=ST.pages[i]; const [x0,y0,x1,y1]=relToAbs(rel,P.w,P.h); const w=Math.max(1,Math.round(x1-x0)), h=Math.max(1,Math.round(y1-y0)); const c=document.createElement('canvas'); c.width=w; c.height=h; c.getContext('2d').drawImage(P.canvas,x0,y0,w,h,0,0,w,h); return c; }
function depthAtY(y,i){ const dp=dpx(i); if(dp.top_y==null||dp.bot_y==null||dp.top_y===dp.bot_y) return null; const base=ST.pageOffsets[i]||0; const td=ST.topFt+base, bd=ST.botFt+base; return td+((y-dp.top_y)/(dp.bot_y-dp.top_y))*(bd-td); }

/* preview */
btnPreview.onclick=()=>{ const r=regs(ST.curr); if(!r['description_col']) return; const P=ST.pages[ST.curr]; const c=crop(ST.curr,r['description_col']); const sb=splitBandsWithParams(c,ST.sepFrac,ST.minBand);
  draw(); const [x0,y0,x1,y1]=relToAbs(r['description_col'],P.w,P.h); ctx.save(); ctx.strokeStyle='#00E5FF'; ctx.lineWidth=1.5; for(const y of sb.seps){ const yy=(y0+y)*ST.zoom; ctx.beginPath(); ctx.moveTo(x0*ST.zoom,yy); ctx.lineTo(x1*ST.zoom,yy); ctx.stroke(); } ctx.restore(); };

/* cfg */
btnSave.onclick=()=>{ const cfg={regions:ST.regions,depthPx:ST.depthPx,topFt:ST.topFt,botFt:ST.botFt,sepFrac:ST.sepFrac,minBand:ST.minBand}; const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(cfg,null,2)],{type:'application/json'})); a.download='borelog_config.json'; a.click(); };
btnLoadCfg.onchange=async e=>{ const f=e.target.files?.[0]; if(!f) return; const cfg=JSON.parse(await f.text()); ST.regions=cfg.regions||{}; ST.depthPx=cfg.depthPx||{}; ST.topFt=cfg.topFt??0; ST.botFt=cfg.botFt??36; ST.sepFrac=cfg.sepFrac??0.45; ST.minBand=cfg.minBand??40; topDepthEl.value=ST.topFt; botDepthEl.value=ST.botFt; sepFracEl.value=ST.sepFrac; minBandEl.value=ST.minBand; draw(); };

/* extract */
btnExtract.onclick=async ()=>{
  extractStatus.textContent='Extracting…';
  // page offsets
  ST.pageOffsets={}; const span=(ST.botFt-ST.topFt); let lastB=null, acc=0;
  for(let i=0;i<ST.pages.length;i++){ let bore=''; const r=regs(i); if(r['bore_box']){ const t=await ocrText(crop(i,r['bore_box']),6); const m=t.match(/([A-Z]{2}-?\d{1,3})/i); if(m)bore=m[1].toUpperCase(); }
    if(!bore && r['header']){ const t=await ocrText(crop(i,r['header']),6); const m=t.match(/\b([A-Z]{2}-?\d{1,3})\b/); if(m)bore=m[1].toUpperCase(); }
    if(lastB===null || bore!==lastB){ acc=0; lastB=bore; } ST.pageOffsets[i]=acc; acc+=span; }

  const rows=[];
  for(let i=0;i<ST.pages.length;i++){
    const r=regs(i); if(!r['description_col']) continue;
    let bore='', lat='', lon='', elev='', water='N/E';
    if(r['header']){ const t=await ocrText(crop(i,r['header']),6); bore=(t.match(/\b([A-Z]{2}-?\d{1,3})\b/i)||['',''])[1];
      lat=(t.match(/LATITUDE.*?([+\-]?\d+\.\d+)/i)||['',''])[1]; lon=(t.match(/LONGITUDE.*?([+\-]?\d+\.\d+)/i)||['',''])[1]; elev=(t.match(/ELEVATION.*?(\d+\.\d+)/i)||['',''])[1];
      const m=t.match(/DEPTH.*?WATER.*?(?:INITIAL.*?([\d.]+))?(?:.*?AFTER.*?24.*?HOURS.*?([\d.]+))?/i); if(m) water=m[1]||m[2]||'N/E'; }
    if(r['bore_box']){ const t=await ocrText(crop(i,r['bore_box']),6); const m=t.match(/([A-Z]{2}-?\d{1,3})/i); if(m) bore=m[1]; }
    if(r['lat_box']){ const t=await ocrText(crop(i,r['lat_box']),6); const m=t.match(/([+\-]?\d+\.\d+)/); if(m) lat=m[1]; }
    if(r['lon_box']){ const t=await ocrText(crop(i,r['lon_box']),6); const m=t.match(/([+\-]?\d+\.\d+)/); if(m) lon=m[1]; }
    if(r['elev_box']){ const t=await ocrText(crop(i,r['elev_box']),6); const m=t.match(/(\d+\.\d+)/); if(m) elev=m[1]; }
    if(r['water_box']){ const t=await ocrText(crop(i,r['water_box']),6); const m=t.match(/(\d+\.\d+)/); if(m) water=m[1]; }

    const descC=crop(i,r['description_col']); let {bands}=splitBandsWithParams(descC,ST.sepFrac,ST.minBand);
    const dp=dpx(i); const [rx0,ry0,rx1,ry1]=relToAbs(r['description_col'],ST.pages[i].w,ST.pages[i].h);
    if(ST.clipDepth && dp.top_y!=null && dp.bot_y!=null){ const b2=[]; for(const [a,b] of bands){ const A=ry0+a, B=ry0+b; const AA=Math.max(A,dp.top_y), BB=Math.min(B,dp.bot_y); if(BB-AA>Math.max(6,ST.minBand/2)) b2.push([AA-ry0, BB-ry0]); } bands=b2; }

    const blocks=[];
    for(const [y0,y1] of bands){ const tmp=document.createElement('canvas'); tmp.width=descC.width; tmp.height=Math.max(1,y1-y0);
      tmp.getContext('2d').drawImage(descC,0,y0,descC.width,tmp.height,0,0,tmp.width,tmp.height);
      const t=(await ocrText(tmp,4)).replace(/[|]/g,' ').trim(); if(t && !/^Description$/i.test(t)) blocks.push({y0_abs:ry0+y0, y1_abs:ry0+y1, text:t, nvals:[]}); }

    if(r['nvalue_col']){ const nC=crop(i,r['nvalue_col']); const nums=await ocrDigitsWithPos(nC); const [nx0,ny0]=relToAbs(r['nvalue_col'],ST.pages[i].w,ST.pages[i].h);
      for(const blk of blocks){ const vals=[]; for(const n of nums){ const cy=ny0+n.y+n.h/2; if(cy>=blk.y0_abs && cy<=blk.y1_abs) vals.push(n.val); }
        const uniq=[]; for(const v of vals){ if(!uniq.length || uniq[uniq.length-1]!==v) uniq.push(v); } blk.nvals=uniq; } }

    for(let k=0;k<blocks.length;k++){ const b=blocks[k]; let y0a=b.y0_abs, y1a=b.y1_abs; const dp2=dpx(i); if(k===0 && ST.snapTop && dp2.top_y!=null) y0a=dp2.top_y; if(k===blocks.length-1 && ST.snapBot && dp2.bot_y!=null) y1a=dp2.bot_y;
      const d0=depthAtY(y0a,i), d1=depthAtY(y1a,i); const from=d0!=null?+d0.toFixed(1):""; const to=d1!=null?+d1.toFixed(1):"";
      const elevFrom=(elev && from!=="")?+(parseFloat(elev)-from).toFixed(1):""; const elevTo=(elev && to!=="")?+(parseFloat(elev)-to).toFixed(1):"";
      rows.push([bore||"",from,to,(b.nvals||[]).join(", ")||"N/A",b.text,lon||"",lat||"",elev||"",water||"N/E",elevFrom,elevTo]); }
    extractStatus.textContent=`Extracting… ${i+1}/${ST.pages.length}`;
  }
  const ws=XLSX.utils.aoa_to_sheet([["Bore L.","From (ft)","To (ft)","SPT N-Value","Soil Layer Description","Longitude","Latitude","Top Elevation (ft)","Water Table (ft)","Elevation From (ft)","Elevation To (ft)"], ...rows]);
  const wb=XLSX.utils.book_new(); XLSX.utils.book_append_sheet(wb, ws, "Bore Logs"); XLSX.writeFile(wb, "bore_logs_v5.xlsx"); extractStatus.textContent=`Done. Rows: ${rows.length}`;
};
