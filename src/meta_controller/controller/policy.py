from __future__ import annotations

from typing import Iterable, List


class MetaPolicy:
    def act(self, observation: Iterable[float]) -> List[float]:
        values = list(observation)
        load = values[1]
        cpu_temp = values[2]
        rssi = values[3]
        rtt = values[4]
        risk_shift = max(-1.0, min(1.0, (cpu_temp - 0.5) - rssi * 0.2))
        return [
            risk_shift,
            max(-1.0, min(1.0, 0.2 - load)),
            max(-1.0, min(1.0, 0.8 - cpu_temp)),
            max(-1.0, min(1.0, rssi - 0.5)),
            max(-1.0, min(1.0, 0.5 - rtt)),
        ]
