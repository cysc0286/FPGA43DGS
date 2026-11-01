// Exercise an explicitly supplied camera-preview URL with a fresh browser profile.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
(async () => {
  const [url, out] = process.argv.slice(2);
  fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({headless: true, channel: process.env.BROWSER_CHANNEL || 'msedge'});
  try {
    const page = await browser.newPage({viewport: {width: 1000, height: 800}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(() => document.getElementById('camera').naturalWidth === 1280);
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('采集'));
    const first = await page.locator('#status').innerText();
    await page.locator('#pause').click();
    assert.equal(await page.locator('#pause').innerText(), '继续显示');
    await page.screenshot({path: path.join(out, 'live-camera-preview.png')});
    await page.locator('#pause').click();
    await page.waitForFunction(previous => document.getElementById('status').textContent !== previous, first);
    await page.setViewportSize({width: 320, height: 720});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    const status = await page.locator('#status').innerText();
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(out, 'preview_browser.json'), JSON.stringify({passed: true, first, status, errors,
      checks: ['live camera image', 'pause', 'resume', '320px layout'],
      scope: 'real board USB camera via local SSH forward; no 3DGS inference'}, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
