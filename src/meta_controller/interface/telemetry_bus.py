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
        self._random = random.Random(seed)
        self._frames = self._make_frames()
        self._index = 0

    def _make_frames(self) -> List[TelemetryFrame]:
        frames = []
        soc = 0.95
        for step in range(32):
            load = min(0.95, 0.35 + 0.02 * step + self._random.uniform(-0.03, 0.03))
            cpu_temp = min(0.95, 0.45 + 0.015 * step + self._random.uniform(-0.02, 0.02))
            rssi = max(0.1, 0.8 - 0.015 * step + self._random.uniform(-0.03, 0.03))
            rtt = min(0.95, 0.2 + 0.01 * step + self._random.uniform(-0.02, 0.02))
            frames.append(TelemetryFrame(soc=max(0.1, soc), load=load, cpu_temp=cpu_temp, rssi=rssi, rtt=rtt))
            soc -= 0.015
        return frames

    def next_frame(self) -> TelemetryFrame:
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return frame
