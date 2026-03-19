from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from meta_controller.controller.action_mapping import ParameterSnapshot


@dataclass
class SchedulerSummary:
    scheduled: int
    succeeded: int
    latency_ms: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "scheduled": self.scheduled,
            "succeeded": self.succeeded,
            "latency_ms": self.latency_ms,
        }


class UTAAClient:
    def schedule(self, observation: list[float], parameters: ParameterSnapshot) -> SchedulerSummary:
        soc, load, cpu_temp, rssi, rtt, _, _ = observation                       # 1. 解析观测值：只取前5个核心指标（剩余2个未使用）
        quality = max(0.0, min(1.0, (soc + (1.0 - load) + rssi + (1.0 - rtt)) / 4.0))   # 2. 计算任务执行质量分（归一化到0-1之间）  # 逻辑：电量+低负载+强信号+低时延 四项平均，
        risk_penalty = max(0.0, parameters.risk_budget - 0.1)             # 3. 计算风险惩罚系数：风险预算超过0.1时才产生惩罚（避免无意义的小值）
        success_ratio = max(0.0, min(1.0, quality - risk_penalty * 0.15 - max(0.0, cpu_temp - 0.7)))   # 4. 计算任务成功概率 质量分 - 风险惩罚 - CPU高温惩罚（温度超过0.7才惩罚）
        scheduled = max(1, int(round(4 + parameters.weights[0])))             # 5. 计算计划调度的任务数：基础4个，结合权重调整，最少1个
        succeeded = int(round(scheduled * success_ratio))                    # 6. 计算实际成功完成的任务数：计划数 × 成功概率（取整）
        latency_ms = 2.0 + parameters.weights[1] * 0.2                     # 7. 计算调度延迟（毫秒）：基础2ms，结合权重调整
        return SchedulerSummary(scheduled=scheduled, succeeded=succeeded, latency_ms=latency_ms)    # 8. 返回调度结果汇总
