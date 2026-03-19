from __future__ import annotations

from typing import Iterable, List


class MetaPolicy:   #当前是规则式策略接口，形式上对应策略函数
    def act(self, observation: Iterable[float]) -> List[float]:
        values = list(observation)         # 拆解观测状态
        load = values[1]
        cpu_temp = values[2]
        rssi = values[3]            # 信号强度
        rtt = values[4]       # 时延
        risk_shift = max(-1.0, min(1.0, (cpu_temp - 0.5) - rssi * 0.2))     # 计算风险偏移量，限制在[-1,1]
        return [           # 输出5维动作，全部归一化到[-1,1]  后续映射为调度参数快照供环境执行
            risk_shift,
            max(-1.0, min(1.0, 0.2 - load)),      # 先给数值「封顶」  如果「计算值」≤ 1.0 → 保留原计算值   如果「计算值」> 1.0 → 强制变成 1.0（不让超过上限）
            max(-1.0, min(1.0, 0.8 - cpu_temp)),   #第二步：max (-1.0, 第一步结果) → 再给数值「保底」  如果第一步结果 ≥ -1.0 → 保留第一步结果  如果第一步结果 < -1.0 → 强制变成 -1.0（不让低于下限）。
            max(-1.0, min(1.0, rssi - 0.5)),    #假设 load = 0.5 → 计算值 = 0.2 - 0.5 = -0.3
            max(-1.0, min(1.0, 0.5 - rtt)),       #假设 load = -1.0（负载极低，几乎无任务）→ 计算值 = 0.2 - (-1.0) = 1.2   #假设 load = 2.0（负载极高，系统满负荷）→ 计算值 = 0.2 - 2.0 = -1.8
        ]
