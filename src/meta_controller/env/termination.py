from __future__ import annotations


def should_terminate(step: int, horizon: int, soc: float, cpu_temp: float, rssi: float) -> bool:
    if horizon > 0 and step >= horizon:
        return True
    if soc <= 0.1:
        return True
    if cpu_temp >= 60.0:
        return True
    if rssi <= 0.05:
        return True
    return False
