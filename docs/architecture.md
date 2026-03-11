# Architecture

## Bi-level control plane

- `runtime/sidecar.py` runs the DRL meta-controller on a slow control loop.
- `interface/parameter_store.py` publishes validated UTAA parameters as atomic snapshots.
- `interface/scheduler_bridge.py` exposes non-blocking reads for the deterministic scheduler.
- `env/edge_env.py` converts telemetry and execution summaries into RL transitions.
- `metrics/collector.py` records every control step and evaluation episode under a shared schema.

## Safety boundary

- UTAA reads only the latest valid parameter snapshot.
- Invalid, stale, or out-of-range meta-controller outputs are rejected.
- A watchdog can force fallback to default safe parameters.
- Scheduling latency remains bounded by store reads, not policy inference.
