"""Bounded UVC exposure controls. No image postprocessing or GPIO access."""
import re
import subprocess

ALLOWED = {'exposure_auto', 'exposure_absolute', 'gain', 'brightness'}


def parse_controls(text):
    controls, current = {}, None
    for line in text.splitlines():
        header = re.match(r'^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$', line)
        if header:
            name, kind, properties = header.groups()
            current = None
            if name not in ALLOWED:
                continue
            values = {k: int(v) for k, v in re.findall(r'(min|max|step|value)=(-?\d+)', properties)}
            if 'value' not in values:
                continue
            current = dict(type=kind, **values, menu={},
                           flags=properties.split('flags=', 1)[1].strip() if 'flags=' in properties else '')
            controls[name] = current
        elif current is not None:
            menu = re.match(r'^\s+(\d+):\s*(.+)$', line)
            if menu:
                current['menu'][int(menu[1])] = menu[2]
    return controls


class CameraControls:
    def __init__(self, device, runner=subprocess.run):
        self.device, self.runner = str(device), runner

    def _run(self, *args):
        result = self.runner(['v4l2-ctl', '-d', self.device, *args],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=4, check=False)
        if result.returncode:
            raise RuntimeError('相机控制失败：' + result.stdout.strip()[:400])
        return result.stdout

    def read(self):
        controls = parse_controls(self._run('--list-ctrls-menus'))
        menu = controls.get('exposure_auto', {}).get('menu', {})
        manual = 1 if 1 in menu else None
        automatic = next((value for value in (3, 0) if value in menu), None)
        active = controls.get('exposure_auto', {}).get('value')
        mode = 'manual' if manual is not None and active == manual else (
            'auto' if automatic is not None and active == automatic else 'unknown')
        return dict(controls=controls, mode=mode, manual_mode=manual, auto_mode=automatic,
                    exposure_unit_ms=0.1)

    def _write(self, name, value):
        assert name in ALLOWED
        self._run('--set-ctrl', f'{name}={value}')

    def apply(self, updates):
        if (not isinstance(updates, dict) or not updates
                or set(updates) - {'mode', 'exposure_absolute', 'gain', 'brightness'}):
            raise ValueError('只允许设置曝光模式、曝光时间、增益和亮度')
        before = self.read()
        mode = updates.get('mode', before['mode'])
        if mode not in ('manual', 'auto'):
            raise ValueError('请选择手动或自动曝光')
        writes = {}
        if 'mode' in updates:
            value = before['manual_mode'] if mode == 'manual' else before['auto_mode']
            if value is None:
                raise ValueError('这只相机不支持所选曝光模式')
            writes['exposure_auto'] = value
        for name in ('exposure_absolute', 'gain', 'brightness'):
            if name not in updates:
                continue
            if name == 'exposure_absolute' and mode != 'manual':
                raise ValueError('请先选择手动曝光，再设置曝光时间')
            value, ctrl = updates[name], before['controls'].get(name)
            if not ctrl or any(flag in ctrl['flags'] for flag in ('read-only', 'disabled')):
                raise ValueError('相机不支持调整：' + name)
            if (type(value) is not int or 'min' not in ctrl or 'max' not in ctrl
                    or not ctrl['min'] <= value <= ctrl['max']
                    or (value-ctrl['min']) % max(1, ctrl.get('step', 1))):
                raise ValueError('相机参数超出范围或步长：' + name)
            writes[name] = value
        touched = []
        try:
            for name, value in writes.items():
                touched.append(name)
                self._write(name, value)
            after = self.read()
            for name, value in writes.items():
                if after['controls'].get(name, {}).get('value') != value:
                    raise RuntimeError('相机未接受该设置，已尝试恢复原值：' + name)
            return after
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            # Restore exposure while manual, then restore the original mode.
            rollback_error = None
            try:
                if 'exposure_absolute' in touched and before['manual_mode'] is not None:
                    self._write('exposure_auto', before['manual_mode'])
                for name in touched:
                    if name != 'exposure_auto':
                        self._write(name, before['controls'][name]['value'])
                if 'exposure_auto' in touched or 'exposure_absolute' in touched:
                    self._write('exposure_auto', before['controls']['exposure_auto']['value'])
            except (OSError, RuntimeError, subprocess.SubprocessError) as rollback:
                rollback_error = str(rollback)
            if rollback_error:
                raise RuntimeError(f'{exc}；恢复设置失败，请刷新读取实际值：{rollback_error}') from exc
            raise
