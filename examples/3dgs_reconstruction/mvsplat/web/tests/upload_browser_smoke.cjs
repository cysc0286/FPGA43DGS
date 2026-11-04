// Real headless-browser / synthetic HTTP contracts only. No device or model I/O.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {chromium}=require('playwright');

(async()=>{
  const out=process.argv[2];assert(out,'Pass an external evidence output directory');
  fs.mkdirSync(out,{recursive:true});
  const token='synthetic-upload-session';
  const read=name=>fs.readFileSync(path.join(__dirname,'../'+name),'utf8').replace('__TOKEN__',JSON.stringify(token));
  const uploadHTML=read('upload_page.html'),captureHTML=read('recording_page.html');
  const oldPath='/synthetic/upload_old/video.mp4',newPath='/synthetic/upload_new/video.mp4';
  const state={state:'idle',input_kind:'usb_camera',clip:null,upload:null,error:null,maintenance:false,
    elapsed_seconds:0,max_seconds:20,free_mib:512,auto_process:true,
    processing_started_epoch:null,postcapture_elapsed_seconds:0,
    build:{phase:'warmed',preheated:true,input_kind:'usb_camera',mode:'clip',processing_complete:false}};
  const actions=[],errors=[],observed=[],checks=[],layout=[];
  const payload=Buffer.from(Array.from({length:2*1024*1024},(_,i)=>i%251));
  const fileName='横竖屏 环视.mp4';let disconnected=false,endedAt=null,finished=false;
  const server=http.createServer(async(request,response)=>{
    const url=new URL(request.url,'http://127.0.0.1');
    const send=(status,body,type='application/json')=>{response.writeHead(status,{'Content-Type':type});response.end(body);};
    try{
      if(request.method==='GET'&&['/','/upload','/capture'].includes(url.pathname))
        return send(200,url.pathname==='/capture'?captureHTML:uploadHTML,'text/html; charset=utf-8');
      if(request.method==='GET'&&url.pathname==='/api/status'){
        if(disconnected)return send(503,JSON.stringify({error:'synthetic connection interruption'}));
        if(endedAt!==null&&!finished)state.postcapture_elapsed_seconds=(Date.now()-endedAt)/1000;
        return send(200,JSON.stringify(state));
      }
      if(request.method==='POST'){
        assert.equal(request.headers.authorization,'Bearer '+token);actions.push(url.pathname);
        if(url.pathname==='/api/prepare/upload'){
          state.build={phase:'initializing',preheated:false,input_kind:'uploaded_video',mode:'clip',processing_complete:false};
          setTimeout(()=>{state.build={...state.build,phase:'warmed',preheated:true};},250);
          return send(202,'{}');
        }
        if(url.pathname==='/api/upload'){
          const chunks=[];for await(const chunk of request)chunks.push(chunk);const body=Buffer.concat(chunks);
          assert.equal(url.searchParams.get('filename'),fileName);
          assert.equal(request.headers['content-type'],'video/mp4');assert.deepEqual(body,payload,'raw body must preserve original bytes');
          observed.push({kind:'raw_upload',bytes:body.length,filename:url.searchParams.get('filename'),
            authorization:'Bearer synthetic token',multipart:false});
          endedAt=Date.now();state.input_kind='uploaded_video';state.processing_started_epoch=endedAt/1000;
          state.upload={complete:true,path:newPath,source_kind:'uploaded_video',original_filename:fileName,
            actual:{width:720,height:1280},duration_seconds:3.,frames:90};
          return send(202,JSON.stringify({upload:state.upload}));
        }
        return send(404,JSON.stringify({error:'unexpected action'}));
      }
      if(request.method==='GET'&&url.pathname==='/frame.jpg')return send(503,'Synthetic test has no USB camera','text/plain');
      if(request.method==='GET'&&url.pathname==='/viewer')return send(200,'<!doctype html><html lang="zh-CN"><body><h1>合成查看器占位</h1><p>此页不包含真实 Gaussian 或 FPGA 结果。</p><script>parent.postMessage({type:"fpga43dgs-viewer-size",height:300},location.origin)</script></body></html>','text/html');
      return send(404,'not found','text/plain');
    }catch(error){errors.push(error.stack||error.message);return send(500,JSON.stringify({error:error.message}));}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const origin='http://camera.test:'+server.address().port;
  let browser;
  try{
    // Route interception bypasses Chromium's native upload progress. A real
    // loopback-only stub plus a test host mapping exercises the actual XHR API.
    browser=await chromium.launch({headless:true,channel:process.env.BROWSER_CHANNEL||'msedge',
      args:['--host-resolver-rules=MAP camera.test 127.0.0.1','--no-proxy-server']});
    const page=await browser.newPage({viewport:{width:1360,height:1000}});
    page.on('pageerror',error=>errors.push(error.message));
    await page.addInitScript(()=>{
      window.syntheticUploadEvents=[];
      const Original=window.XMLHttpRequest;
      window.XMLHttpRequest=class extends Original{
        constructor(){super();this.upload.addEventListener('progress',event=>window.syntheticUploadEvents.push({
          loaded:event.loaded,total:event.total,lengthComputable:event.lengthComputable}));}
      };
    });
    await page.goto(origin+'/upload');
    await page.locator('#workflow-state').filter({hasText:'等待上传'}).waitFor();
    assert(await page.locator('#submit').isDisabled());assert(!(await page.locator('#prepare').isDisabled()));
    assert.equal(await page.locator('#processing-time').innerText(),'—');
    await page.waitForTimeout(600);assert.deepEqual(actions,[],'preheat must not be automatically requested');
    await page.locator('#prepare').click();
    await page.locator('#prepare').filter({hasText:'正在预热'}).waitFor();
    assert.equal(await page.locator('#processing-time').innerText(),'—','preheat is outside post-upload clock');
    state.build.phase='warmed';state.build.preheated=true;
    await page.locator('#prepare').filter({hasText:'上传环境已预热'}).waitFor();
    assert(await page.locator('#prepare').isDisabled());checks.push('manual source-specific preheat; no automatic POST; warmup excluded');
    await page.screenshot({path:path.join(out,'synthetic-upload-desktop-empty.png'),fullPage:true});

    Object.assign(state,{input_kind:'uploaded_video',processing_started_epoch:1,postcapture_elapsed_seconds:5,
      upload:{complete:true,path:oldPath,source_kind:'uploaded_video',original_filename:'old.mp4',
        actual:{width:1920,height:1080},duration_seconds:2.,frames:60},
      build:{phase:'ready',preheated:true,input_kind:'uploaded_video',mode:'clip',clip_path:oldPath,
        processing_complete:true,matches_clip:true,url:origin+'/viewer#synthetic-old'}});
    await page.locator('#result-viewer').waitFor({state:'visible'});
    await page.locator('#video-file').setInputFiles({name:fileName,mimeType:'video/mp4',buffer:payload});
    assert.match(await page.locator('#file-meta').innerText(),/环视/);
    await page.locator('#submit').click();
    await page.waitForFunction(()=>document.getElementById('workflow-state').textContent==='处理中');
    assert(await page.locator('#result-viewer').isHidden(),'old build must not publish new upload');
    assert.equal(await page.locator('#upload-progress').evaluate(element=>element.value),100);
    assert.match(await page.locator('#source-size').innerText(),/720×1280/);
    const uploadEvents=await page.evaluate(()=>window.syntheticUploadEvents);
    assert(uploadEvents.length>0,'XHR must emit native upload progress events: '+JSON.stringify(uploadEvents));
    assert(uploadEvents.some(e=>e.lengthComputable&&e.loaded===payload.length&&e.total===payload.length),
      'native XHR upload progress must measure the raw file: '+JSON.stringify(uploadEvents));
    observed.push({kind:'native_upload_progress',events:uploadEvents});
    const beforeStale=await page.locator('#processing-time').innerText();await page.waitForTimeout(750);
    assert.notEqual(await page.locator('#processing-time').innerText(),beforeStale,'old ready build cannot freeze new timer');
    checks.push('native XHR progress and raw/Bearer bytes','new path/old completed build remains processing');

    disconnected=true;await page.locator('#connection').filter({hasText:'计时继续'}).waitFor();
    const beforeDisconnect=await page.locator('#processing-time').innerText();await page.waitForTimeout(850);
    assert.notEqual(await page.locator('#processing-time').innerText(),beforeDisconnect,'transient disconnect must keep processing clock alive');
    assert(await page.locator('#submit').isDisabled());
    await page.screenshot({path:path.join(out,'synthetic-upload-desktop-disconnected.png'),fullPage:true});
    checks.push('transient network failure keeps timer running and disables resubmit');
    disconnected=false;state.build={phase:'reconstructing',preheated:true,input_kind:'uploaded_video',mode:'clip',
      clip_path:newPath,matches_clip:true,processing_complete:false,url:origin+'/viewer#synthetic-new',
      progress:{stage:'inference',pair:1,of:2}};
    await page.locator('#progress-detail').filter({hasText:'帧对 1 / 2'}).waitFor();
    assert(await page.locator('#result-viewer').isHidden(),'a viewer URL alone must not qualify completion');
    state.build.phase='ready';await page.waitForTimeout(650);
    assert(await page.locator('#result-viewer').isHidden(),'ready without FRAME_COMPLETE remains hidden');
    checks.push('URL and ready alone cannot publish first frame');
    finished=true;state.postcapture_elapsed_seconds=7.25;state.build.processing_complete=true;
    await page.waitForFunction(()=>document.getElementById('workflow-state').textContent==='场景就绪');
    await page.frameLocator('#result-viewer').locator('h1').waitFor();
    assert.equal(await page.locator('#result-viewer').getAttribute('src'),state.build.url);
    const finalTimer=await page.locator('#processing-time').innerText();await page.waitForTimeout(750);
    assert.equal(await page.locator('#processing-time').innerText(),finalTimer);
    checks.push('bound FRAME_COMPLETE publishes existing viewer and freezes timer');

    async function measure(label){
      const details=await page.evaluate(()=>({viewport:innerWidth,scroll:document.documentElement.scrollWidth,
        cards:Array.from(document.querySelectorAll('.workspace>.card')).map(e=>{const r=e.getBoundingClientRect();return{x:r.x,y:r.y,width:r.width,height:r.height}}),
        navigation:Array.from(document.querySelectorAll('nav a')).map(e=>({text:e.textContent,href:e.getAttribute('href'),current:e.getAttribute('aria-current')})),
        buttons:Array.from(document.querySelectorAll('.controls button')).filter(e=>e.getClientRects().length).map(e=>({text:e.textContent,height:e.getBoundingClientRect().height}))}));
      assert(details.scroll<=details.viewport,label+' must not overflow horizontally');layout.push({label,...details});return details;
    }
    const desktop=await measure('upload desktop1360');assert(desktop.cards[0].x<desktop.cards[1].x);
    assert.equal(desktop.navigation.length,2);assert.deepEqual(desktop.navigation.map(e=>e.href),['/capture','/upload']);
    assert(desktop.buttons.every(e=>e.height>=44));
    await page.screenshot({path:path.join(out,'synthetic-upload-desktop-ready.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});const mobile=await measure('upload mobile390');
    assert(mobile.cards[0].y+mobile.cards[0].height<=mobile.cards[1].y,'mobile cards must stack without overlap');
    await page.screenshot({path:path.join(out,'synthetic-upload-mobile390-ready.png'),fullPage:true});
    await page.locator('nav a[href="/capture"]').click();
    await page.locator('h1').filter({hasText:'摄像头采集'}).waitFor();await page.waitForTimeout(650);
    const captureMobile=await measure('capture mobile390');assert.equal(captureMobile.navigation[0].current,'page');
    assert(await page.locator('#result-viewer').isHidden(),'USB page must not claim upload result');
    assert.match(await page.locator('#build-status').innerText(),/电脑视频任务/);
    assert.equal(await page.locator('#progress-detail').innerText(),'');
    assert.equal(await page.locator('#workflow-state').innerText(),'电脑视频任务');
    assert(await page.locator('#camera').isHidden(),'no broken preview before the first valid camera frame');
    await page.screenshot({path:path.join(out,'synthetic-capture-mobile390.png'),fullPage:true});
    await page.setViewportSize({width:1360,height:1000});await measure('capture desktop1360');
    await page.screenshot({path:path.join(out,'synthetic-capture-desktop.png'),fullPage:true});
    await page.locator('nav a[href="/upload"]').click();await page.locator('h1').filter({hasText:'本地视频上传'}).waitFor();
    checks.push('desktop/mobile layout; both page navigation; USB page does not claim upload scene');
    assert.deepEqual(actions,['/api/prepare/upload','/api/upload']);assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({passed:true,scope:'Real headless Edge on camera.test mapped to a loopback-only HTTP stub; synthetic state only, no board, USB, NPU, FPGA, scene quality or real transfer throughput claims',
      browser:browser.version(),errors,actions,checks,observed,layout,final_timer:finalTimer},null,2));
    console.log(JSON.stringify({passed:true,checks:checks.length,actions,output:out}));
  }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1});
