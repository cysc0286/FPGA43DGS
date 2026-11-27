/* Camera writes happen only on explicit Apply; polling never changes settings. */
(() => {
  const byId = id => document.getElementById(id);
  const mode = byId('exposure-mode'), exposure = byId('exposure-ms');
  const gain = byId('camera-gain'), brightness = byId('camera-brightness');
  const apply = byId('camera-controls-apply'), read = byId('camera-controls-read');
  const message = byId('camera-controls-status'), current = byId('camera-controls-current');
  let settings = null, dirty = false, pending = false, captureLocked = false;

  function lock() {
    const disabled = pending || captureLocked || !settings || !settings.editable;
    mode.disabled = disabled || settings.manual_mode === null || settings.auto_mode === null;
    exposure.disabled = disabled || mode.value !== 'manual' || !settings.controls.exposure_absolute;
    gain.disabled = disabled || !settings.controls.gain;
    brightness.disabled = disabled || !settings.controls.brightness;
    apply.disabled = disabled;
    read.disabled = pending;
    const budget = settings ? 1000 / settings.requested_fps : 0;
    byId('camera-controls-hint').textContent = settings && mode.value === 'manual' && Number(exposure.value) > budget
      ? `曝光时间超过当前 ${settings.requested_fps} fps 的单帧间隔（${budget.toFixed(1)} ms），可能降帧或拖影，录像帧率检查可能不通过。`
      : '增加曝光可能带来拖影和帧率下降；增益过高会增加噪点。设置稳定后再录制。';
  }
  window.updateCameraControlsLock = value => { captureLocked = value; lock(); };

  function show(data, overwrite) {
    settings = data;
    const controls = data.controls;
    const exp = controls.exposure_absolute;
    byId('exposure-field').hidden = !exp;
    byId('gain-field').hidden = !controls.gain;
    byId('brightness-field').hidden = !controls.brightness;
    if (overwrite || !dirty) {
      mode.value = data.mode;
      for (const [input, ctrl, factor] of [[exposure, exp, 0.1], [gain, controls.gain, 1], [brightness, controls.brightness, 1]]) {
        if (!ctrl) continue;
        input.min = (ctrl.min * factor).toFixed(factor === 1 ? 0 : 1);
        input.max = (ctrl.max * factor).toFixed(factor === 1 ? 0 : 1);
        input.step = ((ctrl.step || 1) * factor).toFixed(factor === 1 ? 0 : 1);
        input.value = (ctrl.value * factor).toFixed(factor === 1 ? 0 : 1);
      }
    }
    current.textContent = '相机读回：' + (data.mode === 'manual' ? '手动曝光' : data.mode === 'auto' ? '自动曝光' : '未知模式')
      + (exp ? ` · 曝光 ${(exp.value / 10).toFixed(1)} ms${data.mode === 'auto' ? '（自动模式下该寄存器不一定代表实时曝光）' : ''}` : '')
      + (controls.gain ? ` · 增益 ${controls.gain.value}` : '')
      + (controls.brightness ? ` · 亮度 ${controls.brightness.value}` : '');
    lock();
  }

  async function refresh(overwrite = false) {
    if (pending) return;
    pending = true; lock();
    try {
      const response = await fetch('/api/camera-controls', {cache: 'no-store'});
      const data = await response.json();
      if (!response.ok) throw Error(data.error || '无法读取相机设置');
      if (overwrite) dirty = false;
      show(data, overwrite);
    } catch (error) { settings = null; message.textContent = error.message; }
    finally { pending = false; lock(); }
  }
  for (const input of [mode, exposure, gain, brightness]) input.addEventListener('input', () => {
    dirty = true; message.textContent = '设置尚未应用'; lock();
  });
  read.onclick = () => { message.textContent = ''; refresh(true); };
  apply.onclick = async () => {
    if (apply.disabled) return;
    const updates = {mode: mode.value};
    for (const input of [exposure, gain, brightness]) {
      if (!input.disabled && !input.reportValidity()) return;
    }
    if (!exposure.disabled) updates.exposure_absolute = Math.round(Number(exposure.value) * 10);
    if (!gain.disabled) updates.gain = Number(gain.value);
    if (!brightness.disabled) updates.brightness = Number(brightness.value);
    pending = true; lock(); message.textContent = '正在应用并读取确认…';
    try {
      const response = await fetch('/api/camera-controls', {method: 'POST',
        headers: {'X-Session-Token': token, 'Content-Type': 'application/json'}, body: JSON.stringify(updates)});
      const data = await response.json();
      if (!response.ok) throw Error(data.error || '相机设置失败');
      dirty = false; show(data, true); message.textContent = '已应用到相机并核对读回值';
    } catch (error) { message.textContent = error.message; }
    finally { pending = false; lock(); }
  };
  refresh(); setInterval(() => refresh(), 5000);
})();
