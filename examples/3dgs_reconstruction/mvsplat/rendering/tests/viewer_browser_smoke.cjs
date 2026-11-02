// Run against viewer_smoke_server.py; artifacts are synthetic UI evidence only.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
(async () => {
  const out = process.argv[2];
  const {url} = JSON.parse(fs.readFileSync(path.join(out, 'server.json'), 'utf8'));
  const browser = await chromium.launch({headless: true, ...(process.env.BROWSER_CHANNEL ? {channel: process.env.BROWSER_CHANNEL} : {})});
  try {
    const page = await browser.newPage({viewport: {width: 1360, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const requests=[];
    page.on('request',r=>{if(r.url().endsWith('/render')&&r.method()==='POST')requests.push(r.postDataJSON());});
    await page.goto(url);
    await page.locator('#status').filter({hasText: '原生分辨率'}).waitFor();
    const render = async action => {
      const response = page.waitForResponse(r => r.url().endsWith('/render') && r.request().method() === 'POST');
      await action();
      assert.equal((await response).status(), 200);
    };
    const initial=requests.length;
    await page.locator('#yaw').evaluate(e => {e.value = '10'; e.dispatchEvent(new Event('input'));});
    const box = await page.locator('#selector').boundingBox();
    await page.mouse.move(box.x+100, box.y+100);
    await page.mouse.wheel(0, 100);
    await page.mouse.move(box.x+160,box.y+160);await page.mouse.down();await page.mouse.move(box.x+185,box.y+175);await page.mouse.up();
    await page.waitForTimeout(250);
    assert.equal(requests.length,initial,'selection must not send renders');
    await render(()=>page.locator('#confirm').click());
    await page.locator('#confirm').filter({hasText:'确认渲染'}).waitFor();
    assert.equal(requests.length,initial+1);
    const committed=requests.at(-1);
    assert(committed.yaw>10&&committed.pitch<0&&committed.distance>1);
    const camera=fs.readFileSync(path.join(out,'last_camera.bin'));
    const preview=await page.evaluate(()=>candidatePose());
    const expected=[...Array.from({length:3},(_,i)=>[preview.right[i],preview.down[i],preview.forward[i]]).flat(),...preview.position];
    expected.forEach((v,i)=>assert(Math.abs(v-camera.readDoubleLE(24+8*i))<1e-9,'selector and submitted camera must agree'));
    const left=await page.locator('#selector').boundingBox(),right=await page.locator('#view').boundingBox();
    assert(right.x>left.x+left.width,'desktop is two columns');
    await page.evaluate(()=>scrollTo(0,0));
    await page.screenshot({path:path.join(out,'synthetic-viewer-desktop.png'),fullPage:true});
    await page.locator('#reset').click();
    assert.equal(await page.locator('#yaw').inputValue(), '0');
    assert.equal(await page.locator('#distance').inputValue(), '1');
    assert.equal(requests.length,initial+1,'reset selects only');
    await render(()=>page.locator('#confirm').click());
    await page.reload();
    await page.locator('#status').filter({hasText: '原生分辨率'}).waitFor();
    await page.setViewportSize({width: 320, height: 720});
    await page.evaluate(()=>scrollTo(0,0));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join(out, 'synthetic-viewer-320.png'),fullPage:true});
    await page.locator('#quit').click();
    await page.locator('#status').filter({hasText: '会话已结束'}).waitFor();
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(out, 'browser.json'), JSON.stringify({passed: true, errors,
      checks: ['initial RGB','local yaw/wheel/drag without request','explicit confirm','preview/native camera agreement','desktop two columns',
        'reset without request','reload','320px layout','quit'],
      scope: 'headless browser with synthetic RGB; no camera/model/FPGA'}, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
