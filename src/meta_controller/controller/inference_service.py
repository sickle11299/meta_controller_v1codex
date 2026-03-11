from __future__ import annotations

from typing import Iterable

from meta_controller.controller.action_mapping import ParameterSnapshot, SafeActionMapper
from meta_controller.controller.policy import MetaPolicy


class InferenceService:
    def __init__(self, policy: MetaPolicy, mapper: SafeActionMapper) -> None:
        self.policy = policy
        self.mapper = mapper

    def infer(self, observation: Iterable[float], previous: ParameterSnapshot | None, version: int) -> ParameterSnapshot:
        action = self.policy.act(observation)
        return self.mapper.map_action(action, previous=previous, version=version)
