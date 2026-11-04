// Synthetic UI evidence only: no camera, inference, or FPGA device.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const out=process.argv[2],look=process.argv[3]==='look';
 const {url}=JSON.parse(fs.readFileSync(path.join(out,'server.json'),'utf8'));
 const browser=await chromium.launch({headless:true,...(process.env.BROWSER_CHANNEL?{channel:process.env.BROWSER_CHANNEL}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1360,height:768}}),errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(r.url().endsWith('/render')&&r.method()==='POST')requests.push(r.postDataJSON());});
  await page.goto(url);
  const idle=async()=>{await page.waitForFunction(()=>hasFrame&&!busy&&!autoPending&&renderTimer===null);};
  await idle();
  assert.equal(await page.locator('#view-mode').inputValue(),look?'look':'orbit');
  assert.equal(await page.locator('#distance').isVisible(),!look);
  assert.doesNotMatch(await page.locator('body').innerText(),/水平视锥采样覆盖|最大空缺|目标未达|可三角测量|左侧为简化|选好后确认|平移：/);
  const layouts=[];
  for(const viewport of [{width:1360,height:768},{width:1280,height:720}]){
   await page.setViewportSize(viewport);
   await page.waitForTimeout(80);
   const m=await page.evaluate(()=>{
    const b=s=>{const r=document.querySelector(s).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom};};
    return {left:b('#selector'),right:b('.result-surface'),image:b('#view'),controls:[...document.querySelectorAll('button,select,input')].filter(e=>e.getClientRects().length).map(e=>({id:e.id,...b('#'+e.id)})),height:innerHeight};
   });
   assert(Math.abs(m.left.width-m.left.height)<1&&Math.abs(m.right.width-m.right.height)<1,'both viewports square');
   assert(Math.abs(m.left.y-m.right.y)<1&&Math.abs(m.left.height-m.right.height)<1,'square windows aligned');
   assert(Math.abs(m.image.width-m.right.width)<1&&Math.abs(m.image.height-m.right.height)<1,'native RGB fills the window without side bars');
   for(const c of m.controls)assert(c.y>=0&&c.bottom<=m.height,`control ${c.id} fits one desktop viewport`);
   layouts.push({viewport,...m});
  }
  await page.setViewportSize({width:1360,height:768});
  // A slow reply makes queueing defects reproducible without measuring fake speed.
  let active=0,maxActive=0;
  await page.route('**/render',async route=>{active++;maxActive=Math.max(maxActive,active);try{await new Promise(r=>setTimeout(r,160));const response=await route.fetch();await route.fulfill({response});}finally{active--;}});
  const start=requests.length;
  await page.locator('#yaw').evaluate(e=>{e.value=10;e.dispatchEvent(new Event('input'));});
  await page.waitForTimeout(40);
  await page.locator('#yaw').evaluate(e=>{for(let n=0;n<40;n++){e.value=20+n;e.dispatchEvent(new Event('input'));}});
  await idle();
  assert.equal(requests.at(-1).yaw,59,'latest selection wins');
  assert(requests.length-start<10,'rapid selections are coalesced rather than queued');
  const box=await page.locator('#selector').boundingBox();
  await page.mouse.move(box.x+90,box.y+90);await page.mouse.down();
  await page.mouse.move(box.x+150,box.y+120,{steps:15});await page.mouse.up();await idle();
  const submitted=requests.at(-1),pose=await page.evaluate(()=>candidatePose());
  assert(submitted.yaw>59&&submitted.pitch<0,'drag automatically renders its endpoint');
  assert.equal(maxActive,1,'at most one renderer request in flight');
  const camera=fs.readFileSync(path.join(out,'last_camera.bin'));
  const expected=[...Array.from({length:3},(_,i)=>[pose.right[i],pose.down[i],pose.forward[i]]).flat(),...pose.position];
  expected.forEach((v,i)=>assert(Math.abs(v-camera.readDoubleLE(24+8*i))<1e-9,'displayed endpoint matches native camera'));
  if(look){
   assert.deepEqual(pose.position,[2,3,4]);assert(!Object.hasOwn(submitted,'distance')&&!Object.hasOwn(submitted,'pan_x'));
   const address=new URL(url),token=new URLSearchParams(address.hash.slice(1)).get('token');
   for(const body of [{mode:'orbit'},{mode:'look',distance:1},{mode:'look',pan_x:0}]){
    const r=await page.request.post(address.origin+'/render',{headers:{Authorization:'Bearer '+token},data:body});assert.equal(r.status(),400);
   }
   for(const angle of [-180,-90,0,90,180]){
    await page.locator('#yaw').evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('input'));},angle);await idle();
    assert.equal(requests.at(-1).yaw,angle);
    const bytes=fs.readFileSync(path.join(out,'last_camera.bin'));assert.deepEqual([0,1,2].map(i=>bytes.readDoubleLE(96+8*i)),[2,3,4]);
   }
   await page.locator('#selector').focus();await page.keyboard.press('ArrowRight');await idle();assert.equal(requests.at(-1).yaw,-178);
  }
  await page.locator('#reset').click();await idle();assert.equal(requests.at(-1).yaw,0);assert.equal(requests.at(-1).pitch,0);
  const beforeRefresh=requests.length;await page.locator('#confirm').click();await idle();assert.equal(requests.length,beforeRefresh+1,'manual refresh remains usable');
  await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(out,'synthetic-viewer-desktop.png'),fullPage:true});
  await page.reload();await idle();await page.setViewportSize({width:320,height:720});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'no horizontal overflow on mobile');
  await page.screenshot({path:path.join(out,'synthetic-viewer-320.png'),fullPage:true});
  await page.locator('#quit').click();await page.locator('#status').filter({hasText:'会话已结束'}).waitFor();
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({passed:true,errors,layouts,max_inflight:maxActive,
   checks:['square windows','all desktop controls visible','no RGB side bars','no explanatory notes','automatic drag endpoint','automatic sliders/reset','coalesced latest view','single renderer owner','camera ABI','manual refresh','mobile overflow','reload','quit'],
   scope:'Synthetic UI only; latency is deliberately injected, not performance evidence'},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
