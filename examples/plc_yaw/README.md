# PLC 自动偏航执行环境

在仓库根目录执行接口检查（本机约 3 秒）：

```bash
.venv/bin/python -m WindGym.plc_yaw check
```

这是独立的新环境 `WindGym.plc_yaw.PLCYawEnv`。它根据用户提供的
`automatic-yaw-control.md` 和 `pcm51-yaw-control-reference.plcopen.xml`
复现**正常自动偏航的逻辑指令、制动器、电机输出和重启计时**。
原来的 `WindGym.yaw_alignment` 保留。

本版验证结果和关键执行时间见 [验证记录](VERIFICATION.md)。

**可验证的范围是 Python 模型是否实现该源码快照描述的行为。**
附件没有现场参数表、完整任务声明、原始风向滤波实现或实际 I/O 录波，
所以此环境尚未经过现场 PLC 对照验证，也没有证明 RL 的发电收益。

## 1. 先看八个执行场景

```bash
.venv/bin/python -m WindGym.plc_yaw demo --output runs/plc_yaw/my-demo
```

默认生成八段各 180 秒的轨迹，本机约 5 秒；输出目录必须是新目录。
`normal_cw`、`normal_ccw`、`wind_time_stop`、`pressure_stop`、
`pressure_timeout`、`wind_loss`、`electrical_open`、`mode_exit` 分别展示
正反偏航、三种停止方式、压力超时和退出条件。

每个场景输出：

| 文件 | 用途 |
| --- | --- |
| `execution.svg` | 对比风向角、逻辑指令、电机输出、开闸指令、重启计时 |
| `trace.csv` | 每 100 ms 一行，保留原始 PLC 信号名、机舱角度和计时器状态 |
| `events.csv` | 仅保留状态变化与关键事件，便于检查执行顺序 |
| `summary.json` | 参数、场景、种子、参考文件/实现哈希和执行指标 |

根目录的 `summary.csv` 汇总八个场景。曲线来自模拟，不是现场录波。
所有运行结果在 Git 中忽略。

## 2. 回放 PLC 测量信号

```bash
.venv/bin/python -m WindGym.plc_yaw replay \
  --input examples/plc_yaw/example-inputs.csv \
  --config examples/plc_yaw/example-config.json \
  --output runs/plc_yaw/my-replay
```

示例 CSV 是手工设计的信号序列，不是真实机组数据。回放直接使用 CSV 中的
1 秒、30 秒、**5 分钟**风向均值，不再进行滤波，也不把预测的电机转动反馈到输入。
它用于回答“给定这组 PLC 输入，逻辑、电机和制动器会怎样执行”。

`time_s` 从 0 开始，严格递增，落在扫描边界。首行的三种风向、两种风速、
压力、电缆扭转角都必须提供。后续空单元格沿用上一个值；行间也零阶保持。
最后一行会执行一次扫描。布尔量使用 `0/1/true/false`。
未提供的控制标志采用 `PLCInputs` 默认值，完整初值保存在 `summary.json`。
请按实际录波填写这些标志，尤其是上电完成、有效风和自动模式选择。

## 3. 接入强化学习

```python
import numpy as np
from WindGym.plc_yaw import PLCYawEnv, PLCYawConfig, PLCYawScenario

env = PLCYawEnv(PLCYawConfig(episode_seconds=600))
obs, info = env.reset(
    seed=42,
    options={"scenario": PLCYawScenario(vane_bias_deg=5)},
)
done = False
while not done:
    # 0 = 原 PLC 测量输入；[-1, 1] 映射为默认 ±10° 风向修正。
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0], dtype=np.float32)
    )
    done = terminated or truncated
env.close()
```

动作是**原始风向测量的修正量**，在 1 秒/30 秒/5 分钟均值之前施加。
零动作是原控制器基线；动作无法直接指定电机或制动器输出。
每个 RL 决策默认 1 秒，内部执行十个 100 ms PLC 扫描。

观察量为 31 个值：三种风向均值的正余弦、风速、压力、电缆扭转、
逻辑与物理输出、计时释放标志、外部状态和上次修正量。
内部完整计时、真实偏差、传感器偏置仅在 `info`/轨迹中用于诊断，未作为策略输入。
这是部分可观测问题，后续训练可增加历史窗口或递归策略。

本版没有功率/载荷模型。实验奖励定义为：

```text
reward = -∫ abs(true_yaw_error_deg) / 10 dt
         - travel_cost × actual_yaw_travel_deg
         - start_cost × actual_motor_starts
```

它评估对风误差与实际动作成本，不是 PLC 源码中的奖励，也不能据此宣称发电量提升。
已验证 Stable-Baselines3 PPO 可以在此接口完成短训练；尚未训练或评估优化策略。
运行较长训练前，应先替换真实参数与传感器模型，建立同场景的零动作对照。

## 4. 替换参数与检查来源

[example-config.json](example-config.json) 中所有可调值均是**示例值**。
默认扫描周期根据 `MAIN_ControlSubsystems_100ms` 的名称推定，实际任务周期仍需确认。
完整的参数对应、边界规则和假设见 [源码映射说明](../../Docs/docs/plc-yaw-environment.md)。

查看用户原始 XML 中某个 POU（只解析文本，不执行 PLC 代码）：

```bash
.venv/bin/python -m WindGym.plc_yaw inspect \
  --xml /path/to/pcm51-yaw-control-reference.plcopen.xml \
  --pou PRG_TriggerYawMotor
```

省略 `--pou` 可列出全部 POU。输出包含 SHA256 和是否匹配实现使用的快照。
运行新环境不依赖微信临时文件路径；参考指纹已保存在包中，原始附件没有复制到仓库。

## 5. 验证

```bash
.venv/bin/python -m pytest -p no:capture tests/test_plc_yaw.py --no-cov -q
```

`-p no:capture` 绕过本机已有的 pytest/readline 初始化崩溃，不跳过测试。
测试覆盖阈值相等边界、独立 T/H 计算、±1° 停止条件、逻辑与物理延时、
开/合闸、三种停机模式、计时并行、测量回放、实际运动积分及 Gymnasium/PPO 接口。
测试通过不等于与目标 CODESYS 运行时逐位一致；现场录波对照是下一阶段的验证依据。
