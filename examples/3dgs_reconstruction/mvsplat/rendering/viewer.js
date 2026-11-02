'use strict';
const token=new URLSearchParams(location.hash.slice(1)).get('token')||'';
const $=id=>document.getElementById(id),canvas=$('view'),ctx=canvas.getContext('2d'),selector=$('selector');
const state={yaw:0,pitch:0,distance:1,pan_x:0,pan_y:0};
let revision=0,busy=false,stopped=false,pointer=null,scene=null,liveVersion=0,hasFrame=false;
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v)),add=(a,b)=>a.map((v,i)=>v+b[i]),mul=(a,s)=>a.map(v=>v*s),sub=(a,b)=>add(a,mul(b,-1));
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>mul(a,1/Math.hypot(...a));
function rotate(v,axis,degrees){const t=degrees*Math.PI/180,c=Math.cos(t),s=Math.sin(t);return add(add(mul(v,c),mul(cross(axis,v),s)),mul(axis,dot(axis,v)*(1-c)));}
function candidatePose(){
  const m=scene.base.c2w,origin=m.slice(0,3).map(r=>r[3]),right=m.slice(0,3).map(r=>r[0]),down=m.slice(0,3).map(r=>r[1]),forward=m.slice(0,3).map(r=>r[2]);
  const f=scene.focus_distance,pivot=add(add(add(origin,mul(forward,f)),mul(right,f*state.pan_x)),mul(down,f*state.pan_y));
  const offset=rotate(rotate(mul(forward,-f),right,state.pitch),down,state.yaw),look=unit(mul(offset,-1)),r=unit(cross(down,look));
  return {position:add(pivot,mul(offset,state.distance)),right:r,down:cross(look,r),forward:look};
}
function preview(){
  const g=selector.getContext('2d'),w=selector.width,h=selector.height;g.fillStyle='#17252c';g.fillRect(0,0,w,h);
  if(!scene||!scene.base){g.fillStyle='#d6e1e7';g.textAlign='center';g.font='16px system-ui';g.fillText('等待场景点云',w/2,h/2);return;}
  const camera=candidatePose(),k=scene.base.intrinsics_normalized,points=scene.points||[];g.fillStyle='#8cc7bf';
  for(const point of points){const d=sub(point,camera.position),z=dot(d,camera.forward);if(z<=1e-5)continue;
    const x=w*(k[0][0]*dot(d,camera.right)/z+k[0][2]),y=h*(k[1][1]*dot(d,camera.down)/z+k[1][2]);if(x>=0&&x<w&&y>=0&&y<h)g.fillRect(x,y,2.5,2.5);}
  g.strokeStyle='#82969e';g.lineWidth=1;g.beginPath();g.moveTo(w/2-9,h/2);g.lineTo(w/2+9,h/2);g.moveTo(w/2,h/2-9);g.lineTo(w/2,h/2+9);g.stroke();
  if(!points.length){g.fillStyle='#d6e1e7';g.textAlign='center';g.font='15px system-ui';g.fillText('场景已就绪；使用拖动或滑块选择视角',w/2,h*.75);}
}
function sync(){for(const key of ['yaw','pitch','distance'])$(key).value=state[key];$('yaw-value').textContent=state.yaw.toFixed(1)+'°';$('pitch-value').textContent=state.pitch.toFixed(1)+'°';$('distance-value').textContent=state.distance.toFixed(2)+' 倍';$('pose-detail').textContent='平移：'+state.pan_x.toFixed(2)+' / '+state.pan_y.toFixed(2);}
function changed(){revision++;sync();preview();$('selection-state').textContent='待确认的观察点';}
async function request(path,body){const r=await fetch(path,{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok)throw Error(await r.text());return r;}
async function draw(){
  if(busy||stopped||!scene||!scene.base)return;
  busy=true;$('confirm').disabled=true;$('confirm').textContent='正在渲染…';const id=revision,submitted={...state},started=performance.now();
  try{const response=await request('/render',{...submitted,request_id:id});
    if(response.headers.get('X-Request-Id')!==String(id))throw Error('返回的视角编号不匹配，请重新确认');
    const image=await createImageBitmap(await response.blob());canvas.width=image.width;canvas.height=image.height;ctx.drawImage(image,0,0);image.close();
    await new Promise(resolve=>requestAnimationFrame(resolve));hasFrame=true;$('empty-result').hidden=true;
    $('status').textContent='原生分辨率 '+canvas.width+'×'+canvas.height+' · 已收到板端画面'+(revision===id?'':'；左侧还有未提交的选择');
    $('render-time').textContent=response.headers.get('X-Render-Ms')+' ms';$('response-time').textContent=(performance.now()-started).toFixed(1)+' ms';
    $('returned-pose').textContent=submitted.yaw.toFixed(1)+'° / '+submitted.pitch.toFixed(1)+'° / '+submitted.distance.toFixed(2)+' 倍';
    $('frame-state').textContent='视角 #'+id+' · 场景 v'+response.headers.get('X-Scene-Version');
    if(revision===id)$('selection-state').textContent='该视角已显示';
  }catch(error){$('status').textContent='渲染未完成：'+error.message+'。上一帧已保留，可重新确认。';}
  finally{busy=false;$('confirm').disabled=stopped;$('confirm').textContent='确认渲染';}
}
async function loadScene(){const response=await fetch('/scene',{headers:{Authorization:'Bearer '+token},cache:'no-store'});if(!response.ok)throw Error(await response.text());scene=await response.json();
  const gen=scene.generation||{};$('pipeline-status').textContent=(gen.npu_executed?'NPU 参与高斯推理':gen.backend==='cpu'?'CPU 高斯推理':'等待高斯推理状态')+' → '+(scene.renderer_backend==='fpga'?'FPGA 渲染':'测试渲染服务');
  $('confirm').disabled=!scene.base||busy||stopped;preview();return scene;}
async function poll(){if(stopped)return;try{const response=await fetch('/state',{headers:{Authorization:'Bearer '+token},cache:'no-store'});if(response.status===404)return;if(!response.ok)throw Error(await response.text());
  const live=await response.json(),labels={warming:'模型预热中',sampling:'等待下一组有效视角',geometry:'估计位姿与点云',inference:'生成高斯模型',publishing:'校验并更新场景',stopped:'采集已结束',error:'更新已停止'};
  $('live-status').textContent=(labels[live.phase]||live.phase)+' · 模型版本 '+live.version+(live.message?' · '+live.message:'');
  if(live.has_scene&&live.version!==liveVersion){liveVersion=live.version;await loadScene();if(!hasFrame&&!busy)await draw();else $('selection-state').textContent='模型已更新，确认后查看';}
 }catch(error){$('live-status').textContent=error.message;}if(!stopped)setTimeout(poll,1500);}
for(const key of ['yaw','pitch','distance'])$(key).addEventListener('input',e=>{state[key]=Number(e.target.value);changed();});
selector.addEventListener('pointerdown',e=>{pointer={id:e.pointerId,x:e.clientX,y:e.clientY};selector.setPointerCapture(e.pointerId);});
selector.addEventListener('pointermove',e=>{if(!pointer||pointer.id!==e.pointerId)return;const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;pointer.x=e.clientX;pointer.y=e.clientY;
  if(e.shiftKey){state.pan_x=clamp(state.pan_x-dx/1000,-.25,.25);state.pan_y=clamp(state.pan_y-dy/1000,-.25,.25);}else{state.yaw=clamp(state.yaw+dx*.15,-30,30);state.pitch=clamp(state.pitch-dy*.15,-20,20);}changed();});
for(const event of ['pointerup','pointercancel','lostpointercapture'])selector.addEventListener(event,()=>pointer=null);
selector.addEventListener('wheel',e=>{e.preventDefault();state.distance=clamp(state.distance*Math.exp(e.deltaY*.001),.6,1.5);changed();},{passive:false});
$('reset').onclick=()=>{Object.assign(state,{yaw:0,pitch:0,distance:1,pan_x:0,pan_y:0});changed();};
$('confirm').onclick=draw;
$('quit').onclick=async()=>{if(busy){$('status').textContent='当前渲染正在完成，请稍后结束会话';return;}stopped=true;$('confirm').disabled=true;try{await request('/quit',{});$('status').textContent='会话已结束';}catch(error){stopped=false;$('confirm').disabled=false;$('status').textContent=error.message;}};
sync();preview();
loadScene().then(async()=>{if(scene.base)await draw();poll();}).catch(error=>{$('status').textContent='连接失败：'+error.message;});
