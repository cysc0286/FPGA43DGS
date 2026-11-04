// Synthetic browser state-contract test: no camera, model, NPU, or FPGA access.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
  const out=process.argv[2];fs.mkdirSync(out,{recursive:true});
  const html=fs.readFileSync(path.join(__dirname,'../recording_page.html'),'utf8').replace('__TOKEN__',JSON.stringify('synthetic-session'));
  const state={state:'idle',elapsed_seconds:0,max_seconds:15,free_mib:512,clip:null,error:null,auto_process:true,
    processing_started_epoch:null,postcapture_elapsed_seconds:0,
    build:{phase:'initializing',url:null,mode:'clip',preheated:false,processing_complete:false,processing_elapsed_seconds:0}};
  let endedAt=null,finished=false;const actions=[],errors=[];
  const browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge'});
  try{
    const page=await browser.newPage({viewport:{width:1360,height:1000}});page.on('pageerror',e=>errors.push(e.message));
    await page.route('http://camera.test/**',async route=>{
      const request=route.request(),url=new URL(request.url()),p=url.pathname;
      if(p==='/')return route.fulfill({contentType:'text/html',body:html});
      if(p==='/api/status'){
        if(endedAt!==null&&!finished)state.postcapture_elapsed_seconds=(Date.now()-endedAt)/1000;
        return route.fulfill({contentType:'application/json',body:JSON.stringify(state)});
      }
      if(request.method()==='POST'){
        assert.equal(request.headers()['x-session-token'],'synthetic-session');actions.push(p);
        if(p==='/api/start'){state.state='starting';state.elapsed_seconds=0;}
        if(p==='/api/stop'){state.state='stopping';endedAt=Date.now();state.processing_started_epoch=endedAt/1000;}
        return route.fulfill({contentType:'application/json',body:'{}'});
      }
      if(p==='/frame.jpg')return route.fulfill({status:503,body:'合成测试未连接摄像头'});
      if(p==='/viewer')return route.fulfill({contentType:'text/html',body:'<h1>合成双栏查看器占位</h1>'});
      return route.fulfill({status:404,body:'not found'});
    });
    await page.goto('http://camera.test/');
    await page.locator('#build-status').filter({hasText:'预热'}).waitFor();assert(await page.locator('#start').isDisabled());
    assert.equal(await page.locator('#processing-time').innerText(),'—');
    state.build.phase='warmed';state.build.preheated=true;
    await page.locator('#start').filter({hasText:'开始录制'}).waitFor();await page.locator('#start').click();
    await page.waitForFunction(()=>document.getElementById('workflow-state').dataset.state==='STARTING');
    assert(await page.locator('#stop').isDisabled(),'preparation must not offer the recording-end action');
    const waiting=await page.locator('#capture-countdown').innerText();
    assert.match(waiting,/15 秒还未开始计时/);
    await page.waitForTimeout(700);
    assert.equal(await page.locator('#capture-countdown').innerText(),waiting,'warmup must not consume capture duration');
    state.state='recording';state.elapsed_seconds=1.2;
    await page.waitForFunction(()=>document.getElementById('workflow-state').dataset.state==='RECORDING');
    assert.match(await page.locator('#capture-countdown').innerText(),/约 1[23]\.\d 秒后自动结束/);
    assert.equal(await page.locator('#processing-time').innerText(),'—','recording does not start scene-preparation timing');
    await page.screenshot({path:path.join(out,'synthetic-recording-countdown-desktop.png'),fullPage:true});
    await page.setViewportSize({width:320,height:720});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.screenshot({path:path.join(out,'synthetic-recording-countdown-320.png'),fullPage:true});
    await page.setViewportSize({width:1360,height:1000});
    await page.locator('#stop').click();
    await page.waitForFunction(()=>document.getElementById('workflow-state').dataset.state==='PROCESSING');
    assert(await page.locator('#result-viewer').isHidden());
    const before=await page.locator('#processing-time').innerText();await page.waitForTimeout(750);
    assert.notEqual(await page.locator('#processing-time').innerText(),before,'stopping/file verification time must keep counting');
    state.state='recorded';state.clip={frames:120,duration_seconds:4,measured_fps:30,stop_label:'手动结束'};
    Object.assign(state.build,{phase:'reconstructing',url:'http://camera.test/viewer#token=synthetic',matches_clip:true,
      progress:{stage:'inference',pair:1,of:2}});
    await page.locator('#progress-detail').filter({hasText:'帧对 1 / 2'}).waitFor();
    assert(await page.locator('#result-viewer').isHidden(),'a URL does not prove FRAME_COMPLETE');
    state.build.phase='ready';
    await page.waitForTimeout(650);assert(await page.locator('#result-viewer').isHidden(),'ready without real completion remains hidden');
    assert.match(await page.locator('#build-status').innerText(),/等待.*首帧/);
    finished=true;state.postcapture_elapsed_seconds=7.25;state.build.processing_complete=true;
    await page.waitForFunction(()=>document.getElementById('workflow-state').dataset.state==='READY');
    assert.equal(await page.locator('#result-viewer').getAttribute('src'),state.build.url);
    const iframe=page.frameLocator('#result-viewer');
    await iframe.locator('h1').waitFor();
    const child=page.frames().find(f=>f.url().includes('/viewer'));
    await child.evaluate(()=>parent.postMessage({type:'fpga43dgs-viewer-size',height:700},'http://camera.test'));
    await page.waitForFunction(()=>document.getElementById('result-viewer').style.height==='700px');
    await page.evaluate(()=>postMessage({type:'fpga43dgs-viewer-size',height:999},location.origin));
    await page.waitForTimeout(50);assert.equal(await page.locator('#result-viewer').evaluate(e=>e.style.height),'700px','only the actual viewer frame may resize');
    const final=await page.locator('#processing-time').innerText();await page.waitForTimeout(750);
    assert.equal(await page.locator('#processing-time').innerText(),final,'timer freezes only after actual completion');
    assert.deepEqual(actions,['/api/start','/api/stop'],'stop must require no manual reconstruct request');
    await page.screenshot({path:path.join(out,'synthetic-camera-desktop.png'),fullPage:true});
    const captureBox=await page.locator('#capture-card').boundingBox(),processingBox=await page.locator('#processing-card').boundingBox();
    assert(captureBox.x<processingBox.x,'capture precedes processing from left to right');
    await page.setViewportSize({width:320,height:720});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.screenshot({path:path.join(out,'synthetic-camera-320.png'),fullPage:true});
    state.processing_started_epoch=null;state.postcapture_elapsed_seconds=0;
    state.build.processing_started_epoch=Date.now()/1000;state.build.processing_elapsed_seconds=12.5;
    await page.waitForFunction(()=>document.getElementById('processing-time').textContent==='00:12.5');
    state.build.phase='error';state.build.error='合成输入校验失败';
    await page.waitForFunction(()=>document.getElementById('workflow-state').dataset.state==='ERROR');
    assert(await page.locator('#result-viewer').isHidden());assert.match(await page.locator('#error').innerText(),/合成输入校验失败/);
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({passed:true,errors,final_timer:final,actions,
      checks:['preheat excluded from processing timer','warmed allows recording','stop automatically processes',
        'starting is not recording','capture countdown excludes preparation','early end unavailable during starting',
        '15-second countdown only during recording','recording countdown fits desktop and 320px',
        'stopping time included','URL and ready alone cannot publish result','FRAME_COMPLETE freezes timer',
        'result embedded as iframe','origin/source-bound iframe resizing','capture before processing','manual retry timer source','error hides result','320px no overflow'],
      scope:'Synthetic HTTP state contract only; not real camera/NPU/FPGA or image-quality evidence'},null,2));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
