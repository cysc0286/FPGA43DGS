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
    const page = await browser.newPage({viewport: {width: 800, height: 800}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.locator('#status').filter({hasText: '原生分辨率'}).waitFor();
    const render = async action => {
      const response = page.waitForResponse(r => r.url().endsWith('/render') && r.request().method() === 'POST');
      await action();
      assert.equal((await response).status(), 200);
    };
    await render(() => page.locator('#yaw').evaluate(e => {e.value = '10'; e.dispatchEvent(new Event('input'));}));
    const box = await page.locator('#view').boundingBox();
    await page.mouse.move(box.x+100, box.y+100);
    await render(() => page.mouse.wheel(0, 100));
    await render(() => page.locator('#reset').click());
    assert.equal(await page.locator('#yaw').inputValue(), '0');
    assert.equal(await page.locator('#distance').inputValue(), '1');
    await page.reload();
    await page.locator('#status').filter({hasText: '原生分辨率'}).waitFor();
    await page.setViewportSize({width: 320, height: 720});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join(out, 'synthetic-viewer-320.png')});
    await page.locator('#quit').click();
    await page.locator('#status').filter({hasText: '会话已结束'}).waitFor();
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(out, 'browser.json'), JSON.stringify({passed: true, errors,
      checks: ['initial RGB', 'yaw', 'wheel zoom', 'reset', 'reload', '320px layout', 'quit'],
      scope: 'headless browser with synthetic RGB; no camera/model/FPGA'}, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
