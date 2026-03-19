from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TelemetryFrame:
    soc: float
    load: float
    cpu_temp: float
    rssi: float
    rtt: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "soc": self.soc,
            "load": self.load,
            "cpu_temp": self.cpu_temp,
            "rssi": self.rssi,
            "rtt": self.rtt,
        }


class TelemetryBus:
    def __init__(self, seed: int = 0) -> None:
        self._random = random.Random(seed)     # 固定随机种子，保证生成的数据可复
        self._frames = self._make_frames()    # 生成32组遥测数据帧
        self._index = 0                       # 记录当前获取数据的索引

    def _make_frames(self) -> List[TelemetryFrame]:
        frames = []
        soc = 0.95       # 初始电量95%
        for step in range(32):           # 循环生成32组数据
            load = min(0.95, 0.35 + 0.02 * step + self._random.uniform(-0.03, 0.03))          # 1. 负载：基础值0.35，每步增加0.02，±0.03随机扰动，上限0.95（逐渐升高） 
            cpu_temp = min(0.95, 0.45 + 0.015 * step + self._random.uniform(-0.02, 0.02))     # 2. CPU温度：基础值0.45，每步增加0.015，±0.02随机扰动，上限0.95（逐渐升高）
            rssi = max(0.1, 0.8 - 0.015 * step + self._random.uniform(-0.03, 0.03))           # 3. 信号强度：基础值0.8，每步减少0.015，±0.03随机扰动，下限0.1（逐渐降低）
            rtt = min(0.95, 0.2 + 0.01 * step + self._random.uniform(-0.02, 0.02))           # 4. 网络时延：基础值0.2，每步增加0.01，±0.02随机扰动，上限0.95（逐渐升高）
            frames.append(TelemetryFrame(soc=max(0.1, soc), load=load, cpu_temp=cpu_temp, rssi=rssi, rtt=rtt))     # 5. 电量：初始0.95，每步减少0.015，下限0.1（逐渐降低）
            soc -= 0.015                 # 每生成一组数据，电量减少0.015
        return frames

    def next_frame(self) -> TelemetryFrame:            #循环获取数据
        frame = self._frames[self._index % len(self._frames)]         # 取模运算实现循环获取（索引超过32后，重新从0开始）
        self._index += 1          # 索引自增
        return frame
