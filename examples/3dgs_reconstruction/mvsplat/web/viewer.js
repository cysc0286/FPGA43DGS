'use strict';
const token=new URLSearchParams(location.hash.slice(1)).get('token')||'';
const $=id=>document.getElementById(id),canvas=$('view'),ctx=canvas.getContext('2d'),selector=$('selector');
const state={mode:'orbit',yaw:0,pitch:0,distance:1,pan_x:0,pan_y:0};
let revision=0,busy=false,stopped=false,pointer=null,scene=null,liveVersion=0,hasFrame=false;
let renderTimer=null,autoPending=false,lastRenderStarted=-Infinity;
let hostViewportHeight=null,layoutPending=false;
const AUTO_INTERVAL_MS=100;
function fitViewports(){
  const panels=document.querySelectorAll('.workspace>section');
  if(!panels.length)return;
  const width=Math.min(...Array.from(panels,p=>p.clientWidth));
  let size=Math.min(width,320);
  if(window.innerWidth>760){
    const box=selector.getBoundingClientRect();
    const controls=Array.from(document.querySelectorAll('button,select,input')).filter(e=>e.getClientRects().length);
    const bottom=Math.max(...controls.map(e=>e.getBoundingClientRect().bottom));
    const tail=Math.max(0,bottom-box.bottom),budget=hostViewportHeight||window.innerHeight;
    size=Math.min(width,Math.max(180,budget-16-box.top-tail));
  }
  size=Math.floor(size);
  if(Math.abs(selector.getBoundingClientRect().width-size)>=1)document.documentElement.style.setProperty('--viewport-size',size+'px');
}
function scheduleLayout(){if(layoutPending)return;layoutPending=true;setTimeout(()=>{layoutPending=false;fitViewports();},0);}
window.addEventListener('resize',scheduleLayout);
window.addEventListener('message',event=>{if(event.source===window.parent&&event.data?.type==='fpga43dgs-viewer-viewport'&&Number.isFinite(event.data.height)){hostViewportHeight=clamp(event.data.height,320,2160);scheduleLayout();}});
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v)),wrap=v=>((v+180)%360+360)%360-180;
const add=(a,b)=>a.map((v,i)=>v+b[i]),mul=(a,s)=>a.map(v=>v*s),sub=(a,b)=>add(a,mul(b,-1));
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>mul(a,1/Math.hypot(...a));
function rotate(v,axis,degrees){const t=degrees*Math.PI/180,c=Math.cos(t),s=Math.sin(t);return add(add(mul(v,c),mul(cross(axis,v),s)),mul(axis,dot(axis,v)*(1-c)));}
function referenceAxes(){const m=scene.base.c2w;return {origin:m.slice(0,3).map(r=>r[3]),right:m.slice(0,3).map(r=>r[0]),down:m.slice(0,3).map(r=>r[1]),forward:m.slice(0,3).map(r=>r[2])};}
function candidatePose(){
  const a=referenceAxes(),turn=v=>rotate(rotate(v,a.right,state.pitch),a.down,state.yaw);
  const right=turn(a.right),down=turn(a.down),forward=turn(a.forward),f=scene.focus_distance;
  const pivot=add(add(add(a.origin,mul(a.forward,f)),mul(a.right,f*state.pan_x)),mul(a.down,f*state.pan_y));
  return {position:state.mode==='look'?[...scene.capture_center]:sub(pivot,mul(forward,f*state.distance)),right,down,forward};
}
function coverageInset(g,camera,w,h){
  const coverage=scene.coverage;if(!coverage||!coverage.directions||!coverage.directions.length)return;
  const a=referenceAxes(),referenceYaw=Math.atan2(a.forward[0],a.forward[2]),cx=w-67,cy=h-67,r=44,toAngle=v=>Math.atan2(v[0],v[2])-referenceYaw-Math.PI/2;
  g.fillStyle='#17252ce8';g.fillRect(w-130,h-144,124,138);
  g.strokeStyle='#4c626b';g.lineWidth=7;g.beginPath();g.arc(cx,cy,r,0,2*Math.PI);g.stroke();
  g.strokeStyle='#8cc7bf';
  if(Array.isArray(coverage.covered_yaw_sample_degrees)){
    for(const degrees of coverage.covered_yaw_sample_degrees){const angle=degrees*Math.PI/180-referenceYaw-Math.PI/2;g.beginPath();g.arc(cx,cy,r,angle-.5*Math.PI/180,angle+.5*Math.PI/180);g.stroke();}
  }else{const half=(coverage.crop_horizontal_fov_degrees||0)*Math.PI/360;
    for(const d of coverage.directions){const angle=toAngle(d);g.beginPath();g.arc(cx,cy,r,angle-half,angle+half);g.stroke();}}
  const angle=toAngle(camera.forward);g.fillStyle='#efc88c';g.strokeStyle='#efc88c';g.lineWidth=2;
  g.beginPath();g.moveTo(cx,cy);g.lineTo(cx+Math.cos(angle)*33,cy+Math.sin(angle)*33);g.stroke();
  g.beginPath();g.arc(cx,cy,4,0,2*Math.PI);g.fill();g.font='11px system-ui';g.textAlign='center';g.fillStyle='#d6e1e7';
  g.fillText('参考 0°',cx,h-126);g.fillText('● 相机 · 绿弧为水平采样',cx,h-10);
}
function preview(){
  const g=selector.getContext('2d'),w=selector.width,h=selector.height;g.fillStyle='#17252c';g.fillRect(0,0,w,h);
  if(!scene||!scene.base){g.fillStyle='#d6e1e7';g.textAlign='center';g.font='16px system-ui';g.fillText('等待场景点云',w/2,h/2);return;}
  const camera=candidatePose(),k=scene.base.intrinsics_normalized,points=scene.points||[];
  // Preserve the rendered camera's aspect ratio inside the wider selector.
  const aspect=scene.base.width/scene.base.height,pw=Math.min(w,h*aspect),ph=pw/aspect,ox=(w-pw)/2,oy=(h-ph)/2;g.fillStyle='#8cc7bf';
  for(const point of points){const d=sub(point,camera.position),z=dot(d,camera.forward);if(z<=1e-5)continue;
    const x=ox+pw*(k[0][0]*dot(d,camera.right)/z+k[0][2]),y=oy+ph*(k[1][1]*dot(d,camera.down)/z+k[1][2]);if(x>=ox&&x<ox+pw&&y>=oy&&y<oy+ph)g.fillRect(x,y,2.5,2.5);}
  g.strokeStyle='#82969e';g.lineWidth=1;g.beginPath();g.moveTo(w/2-9,h/2);g.lineTo(w/2+9,h/2);g.moveTo(w/2,h/2-9);g.lineTo(w/2,h/2+9);g.stroke();
  if(!points.length){g.fillStyle='#d6e1e7';g.textAlign='center';g.font='15px system-ui';g.fillText('场景已就绪；拖动或使用滑块选择方向',w/2,h*.6);}
  coverageInset(g,camera,w,h);
  $('position-detail').textContent='相机 XYZ '+camera.position.map(v=>v.toFixed(2)).join(' / ');
}
function sync(){
  for(const key of ['yaw','pitch','distance'])$(key).value=state[key];$('view-mode').value=state.mode;
  $('yaw-value').textContent=state.yaw.toFixed(1)+'°';$('pitch-value').textContent=state.pitch.toFixed(1)+'°';$('distance-value').textContent=state.distance.toFixed(2)+' 倍';
  const look=state.mode==='look';$('distance').hidden=look;$('distance-label').hidden=look;
  $('pose-detail').textContent=look?'观察位置固定在输入记录的拍摄中心；水平可转一整圈。':'平移：'+state.pan_x.toFixed(2)+' / '+state.pan_y.toFixed(2);
  $('heading-detail').textContent='相对参考方向 '+((state.yaw+360)%360).toFixed(1)+'° · 俯仰 '+state.pitch.toFixed(1)+'°';
  $('interaction-hint').textContent=look?'拖动或使用方向键环视，自动更新画面。':'拖动旋转、滚轮缩放；Shift 拖动平移，自动更新画面。';
}
function queueAutomatic(immediate=false){
  autoPending=true;
  if(stopped||busy||!scene||!scene.base)return;
  if(renderTimer!==null){if(!immediate)return;clearTimeout(renderTimer);}
  renderTimer=setTimeout(()=>{renderTimer=null;if(autoPending&&!busy&&!stopped)draw();},immediate?0:Math.max(0,AUTO_INTERVAL_MS-(performance.now()-lastRenderStarted)));
}
function changed(){revision++;sync();preview();$('selection-state').textContent='自动更新观察方向';queueAutomatic();}
async function request(path,body){const r=await fetch(path,{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok)throw Error(await r.text());return r;}
async function draw(){
  if(busy||stopped||!scene||!scene.base)return;
  if(renderTimer!==null){clearTimeout(renderTimer);renderTimer=null;}autoPending=false;lastRenderStarted=performance.now();
  busy=true;$('confirm').disabled=true;$('confirm').textContent='正在渲染…';const id=revision,submitted={...state},started=performance.now(),payload={...submitted,request_id:id};
  if(submitted.mode==='look')for(const key of ['distance','pan_x','pan_y'])delete payload[key];
  if(!scene.camera_controls)delete payload.mode;
  try{const response=await request('/render',payload);
    if(response.headers.get('X-Request-Id')!==String(id))throw Error('返回的视角编号不匹配，请重新确认');
    const image=await createImageBitmap(await response.blob());canvas.width=image.width;canvas.height=image.height;ctx.drawImage(image,0,0);image.close();
    // Pixel handoff is complete after drawImage. Offscreen iframes may suspend
    // animation frames, so visible-paint scheduling must never lock the request.
    hasFrame=true;$('empty-result').hidden=true;
    $('status').textContent='原生分辨率 '+canvas.width+'×'+canvas.height+' · 板端画面已就绪'+(revision===id?'':'；正在更新至最新方向');
    $('render-time').textContent=response.headers.get('X-Render-Ms')+' ms';$('response-time').textContent=(performance.now()-started).toFixed(1)+' ms';
    $('returned-pose').textContent=(submitted.mode==='look'?'固定中心 · ':'绕场景 · ')+submitted.yaw.toFixed(1)+'° / '+submitted.pitch.toFixed(1)+'°';
    $('frame-state').textContent='视角 #'+id+' · 场景 v'+response.headers.get('X-Scene-Version');
    if(revision===id)$('selection-state').textContent='该视角画面已就绪';
  }catch(error){$('status').textContent='渲染未完成：'+error.message+'。上一帧已保留，可重新确认。';}
  finally{busy=false;$('confirm').disabled=stopped;$('confirm').textContent='刷新画面';if(autoPending&&!stopped)queueAutomatic(pointer===null);}
}
async function loadScene(){const response=await fetch('/scene',{headers:{Authorization:'Bearer '+token},cache:'no-store'});if(!response.ok)throw Error(await response.text());
  const first=scene===null;scene=await response.json();const controls=scene.camera_controls||{},modes=controls.modes||['orbit'];
  for(const option of $('view-mode').options)option.disabled=!modes.includes(option.value);
  if(first||!modes.includes(state.mode)){state.mode=controls.default_mode||'orbit';state.distance=1;state.pan_x=0;state.pan_y=0;}
  for(const key of ['yaw','pitch']){const bounds=controls[key]||(key==='yaw'?[-180,180]:[-85,85]);$(key).min=bounds[0];$(key).max=bounds[1];state[key]=clamp(state[key],...bounds);}
  const gen=scene.generation||{};$('pipeline-status').textContent=(gen.npu_executed?'NPU 参与高斯推理':gen.backend==='cpu'?'CPU 高斯推理':'等待高斯推理状态')+' → '+(scene.renderer_backend==='fpga'?'FPGA 渲染':'测试渲染服务');
  $('confirm').disabled=!scene.base||busy||stopped;sync();preview();return scene;}
async function poll(){if(stopped)return;try{const response=await fetch('/state',{headers:{Authorization:'Bearer '+token},cache:'no-store'});if(response.status===404)return;if(!response.ok)throw Error(await response.text());
  const live=await response.json(),labels={warming:'模型预热中',sampling:'等待下一组有效视角',geometry:'估计位姿与点云',inference:'生成高斯模型',publishing:'校验并更新场景',stopped:'采集已结束',error:'更新已停止'};
  $('live-status').textContent=(labels[live.phase]||live.phase)+' · 模型版本 '+live.version+(live.message?' · '+live.message:'');
  if(live.has_scene&&live.version!==liveVersion){liveVersion=live.version;await loadScene();if(!hasFrame&&!busy)await draw();else $('selection-state').textContent='模型已更新，确认后查看';}
 }catch(error){$('live-status').textContent=error.message;}if(!stopped)setTimeout(poll,1500);}
for(const key of ['yaw','pitch','distance'])$(key).addEventListener('input',e=>{state[key]=Number(e.target.value);changed();});
$('view-mode').onchange=e=>{state.mode=e.target.value;state.distance=1;state.pan_x=0;state.pan_y=0;changed();};
selector.addEventListener('pointerdown',e=>{pointer={id:e.pointerId,x:e.clientX,y:e.clientY};selector.setPointerCapture(e.pointerId);});
selector.addEventListener('pointermove',e=>{if(!pointer||pointer.id!==e.pointerId)return;const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;pointer.x=e.clientX;pointer.y=e.clientY;
  if(e.shiftKey&&state.mode==='orbit'){state.pan_x=clamp(state.pan_x-dx/1000,-.25,.25);state.pan_y=clamp(state.pan_y-dy/1000,-.25,.25);}else{state.yaw=wrap(state.yaw+dx*.3);state.pitch=clamp(state.pitch-dy*.3,-85,85);}changed();});
for(const event of ['pointerup','pointercancel','lostpointercapture'])selector.addEventListener(event,()=>{pointer=null;if(autoPending)queueAutomatic(true);});
selector.addEventListener('wheel',e=>{e.preventDefault();if(state.mode==='look')return;state.distance=clamp(state.distance*Math.exp(e.deltaY*.001),.6,1.5);changed();},{passive:false});
selector.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();const amount=e.shiftKey?10:2;if(e.key==='ArrowLeft')state.yaw=wrap(state.yaw-amount);if(e.key==='ArrowRight')state.yaw=wrap(state.yaw+amount);if(e.key==='ArrowUp')state.pitch=clamp(state.pitch+amount,-85,85);if(e.key==='ArrowDown')state.pitch=clamp(state.pitch-amount,-85,85);changed();});
$('reset').onclick=()=>{Object.assign(state,{yaw:0,pitch:0,distance:1,pan_x:0,pan_y:0});changed();};
$('confirm').onclick=draw;
if(typeof ResizeObserver!=='undefined')new ResizeObserver(()=>{const w=Math.round(selector.clientWidth),h=Math.round(selector.clientHeight);if(w>0&&h>0&&(selector.width!==w||selector.height!==h)){selector.width=w;selector.height=h;preview();}}).observe(selector);
$('quit').onclick=async()=>{if(busy){$('status').textContent='当前渲染正在完成，请稍后结束会话';return;}stopped=true;autoPending=false;if(renderTimer!==null){clearTimeout(renderTimer);renderTimer=null;}$('confirm').disabled=true;try{await request('/quit',{});$('status').textContent='会话已结束';}catch(error){stopped=false;$('confirm').disabled=false;$('status').textContent=error.message;}};
if(typeof ResizeObserver!=='undefined'){let resizePending=false;new ResizeObserver(()=>{scheduleLayout();if(window.parent===window||resizePending)return;resizePending=true;requestAnimationFrame(()=>{resizePending=false;window.parent.postMessage({type:'fpga43dgs-viewer-size',height:Math.ceil(document.body.getBoundingClientRect().height)},'*');});}).observe(document.body);}
sync();preview();loadScene().then(async()=>{if(scene.base)await draw();poll();}).catch(error=>{$('status').textContent='连接失败：'+error.message;});
