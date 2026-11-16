# J17 23/21 相机云台固件

这是2026-11-15已生成的两路PWM整板候选，保留渲染逻辑，新增水平、俯仰控制。硬件实现已完成；实板PWM、电机运动及与渲染共存尚未验收，`default_enabled=false`，不会由测试脚本自动安装。

- `BOOT.bin`：启动镜像，SHA-256为`22c06060f52d590d589fda2cb027cce9a6732c4f243ba68c47d5d70bd2075f35`。
- `ai7030_edif_top_disable_icap.bit`：对应PL位流。
- `manifest.json`：文件大小、SHA-256、RTL/XDC对应关系、工具版本、接线与验证范围。
- `boot_comparison.json`：原构建的分区比较证据；非PL分区保持不变。
- `integration.patch`：相对原厂父平台的顶层寄存器/PWM接线变更。需配套原厂平台，不能单凭补丁从空目录重建全板。
- `reports/`：本候选已有的资源、时序、引脚和DRC报告，不是2026-11-16新上板结果。

水平接J17物理23脚，俯仰接21脚，第2脚共地；Bank 12、LVCMOS33。PWM为50Hz，寄存器时钟100MHz，SRV1 ABI2，俯仰1000–2000µs限位、500ms看门狗。轨迹由ARM程序控制，调整JSON不需要重新综合。

全板Slice为19551/19650、BRAM36等效218.5/265、DSP252/400；setup/hold最差余量0.015/0.029ns。继承的厂商脉宽违例为−0.409ns，未声称全板时序签核。此前RTL的PWM53项、总线路由18项检查通过，实际输出波形和机械角度仍待测。

从仓库根目录执行`python tools/check_camera_firmware.py`核对发布文件。已验收的四路渲染固件仍在`rendering/render_main/hardware/BOOT.bin`，身份为`b4e6d4d238d8a33bbf77c4d18df43143fa4304f7c713f09b479c6d5ccd39f9c6`；两者不能互换验收状态。先按[四步测试](../../acceptance/README.md)完成现场验证。
