from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_frame_temporal_bottleneck_hfteacher_gate_scale_sweep as sweep  # noqa: E402

sweep.LOCAL_GENERATED_ROOT = ROOT / "data" / "frame_temporal_bottleneck_hfteacher_v2_gate_scale_sweep_generated"
sweep.BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "frame_temporal_bottleneck_hfteacher_v2_gate_scale_sweep_seed231_all5"
sweep.SIDE_BY_SIDE_DIR = ROOT / "data" / "frame_temporal_bottleneck_hfteacher_v2_gate_scale_sweep_side_by_side_seed231_all5"
sweep.LOG_DIR = ROOT / "data" / "modal_logs" / "frame_temporal_bottleneck_hfteacher_v2_gate_scale_sweep_seed231_all5"

sweep.RUN_NAME = "ltx2b_dist098_waymo24_frame_temporal_bottleneck_hfteacher_v2_seed231_from_shifted_noaction_step003000_steps1000"
sweep.CHECKPOINT_VOLUME = "ltx2b-dist098-waymo24-framebneck-hft-v2-r16-ckpts"
sweep.ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-framebneck-hft-v2-infer"
sweep.RUNS_ROOT = "distilled098_framebottleneck_hfteacher_v2_action_lora_24fps_minterpolate_seed231_runs"
sweep.WRAPPER = "scripts/wrappers/generate_waymo24_distilled_frame_temporal_bottleneck_hf_teacher_action_minterpolate_lora_v2.py"

sweep.SWEEP_CHECKPOINTS = ["step_000050", "step_000100", "step_000250", "step_000500", "step_001000"]
sweep.COUNTERFACTUAL_BASE_CHECKPOINTS = ["step_000050", "step_000100", "step_000250"]

sweep.GATE_MANIFEST = (
    sweep.BENCHMARK_DIR / "manifest_frame_temporal_bottleneck_hfteacher_v2_gate_scale_sweep_seed231_all5.json"
)
sweep.COUNTERFACTUAL_MANIFEST = (
    sweep.BENCHMARK_DIR / "manifest_frame_temporal_bottleneck_hfteacher_v2_gate_scale_counterfactual_seed231_all5.json"
)
sweep.SELECTION_PATH = sweep.BENCHMARK_DIR / "phase2_v2_gate_scale_selection.json"


if __name__ == "__main__":
    sweep.main()
