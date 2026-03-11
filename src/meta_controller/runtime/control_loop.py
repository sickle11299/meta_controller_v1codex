from __future__ import annotations

from typing import Dict

from meta_controller.controller.action_mapping import ParameterSnapshot
from meta_controller.controller.inference_service import InferenceService
from meta_controller.interface.parameter_store import ParameterStore


def run_control_step(
    inference: InferenceService,
    store: ParameterStore,
    observation: list[float],
    version: int,
) -> ParameterSnapshot:
    previous = store.get_latest()
    snapshot = inference.infer(observation, previous=previous, version=version)
    store.publish(snapshot)
    return snapshot
