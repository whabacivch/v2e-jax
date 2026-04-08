"""v2e_jax — JAX-native streaming DVS emulator."""

from .dvs_core import (
    DVSParams,
    DVSSensorState,
    SparseEvents,
    build_threshold_maps,
    dense_counts_to_sparse_events,
    dvs_init,
    dvs_step,
    make_dvs_step_fn,
    run_dvs_count_scan,
    run_dvs_count_scan_jit,
    run_dvs_dense_scan,
    run_dvs_dense_scan_jit,
)
from .upsample import (
    VALID_SUBFRAME_SCHEDULES,
    choose_adaptive_steps,
    temporal_upsample_adaptive_linear,
    temporal_upsample_linear,
    temporal_upsample_motion_compensated,
    upsample_interval_linear,
    upsample_interval_motion_compensated,
)

__all__ = [
    # params / state
    "DVSParams",
    "DVSSensorState",
    "SparseEvents",
    # threshold maps
    "build_threshold_maps",
    # streaming DVS API
    "dvs_init",
    "dvs_step",
    "make_dvs_step_fn",
    # offline scan API
    "run_dvs_count_scan",
    "run_dvs_count_scan_jit",
    "run_dvs_dense_scan",
    "run_dvs_dense_scan_jit",
    # sparse packing
    "dense_counts_to_sparse_events",
    # upsamplers
    "VALID_SUBFRAME_SCHEDULES",
    "upsample_interval_linear",
    "upsample_interval_motion_compensated",
    "choose_adaptive_steps",
    "temporal_upsample_linear",
    "temporal_upsample_adaptive_linear",
    "temporal_upsample_motion_compensated",
]
