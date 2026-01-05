"""Utils module for tomography processing."""

from .alignment import (
    displace_cont,
    find_error,
    find_min_pos,
    divide_sino_line,
    extract_width_displacement,
    detect_and_fix_misalignment,
)
from .display import (
    print_img_data,
    plot_two_sino_planes,
    generate_views,
    fun_mosaic_prjs,
    plot_k3d_volume,
    save_k3d_volume,
    compare_volume_slices,
    compare_multiple_volumes,
)
from .file import (
    read_raw_img_from_detector,
    read_raw_img_folder,
    save_as_mhd_raw,
    safe_load_npz,
)

from .metric import calculate_snr, calculate_cnr

from .processing import normalize_volume_uint8

__all__ = [
    # Alignment functions
    "displace_cont",
    "find_error",
    "find_min_pos",
    "divide_sino_line",
    "extract_width_displacement",
    "detect_and_fix_misalignment",
    # Display functions
    "print_img_data",
    "plot_two_sino_planes",
    "generate_views",
    "fun_mosaic_prjs",
    "plot_k3d_volume",
    "save_k3d_volume",
    "compare_volume_slices",
    "compare_multiple_volumes",
    # File I/O functions
    "read_raw_img_from_detector",
    "read_raw_img_folder",
    "save_as_mhd_raw",
    "safe_load_npz",
    # Processing functions
    "normalize_volume_uint8",
    # Metric functions
    "calculate_snr",
    "calculate_cnr",
]
