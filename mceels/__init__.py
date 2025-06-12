"""
MCEELS Utility Functions

This module provides utility functions for EELS (Electron Energy Loss Spectroscopy) data analysis,
including preprocessing, background subtraction, Gaussian fitting, peak analysis, and plotting.
"""

# Import all functions from submodules
from .preprocessing import (
    normalise, shift, get_x_array, get_closest_x_index,
    gauss_blur_data, preprocess, rotate
)
from .background import (
    create_element_config,
    process_nmc811_elements,
    improved_background_removal,
    process_multiple_elements,
)
from .gaussian import (
    gauss, double_gauss, triple_gauss,
    fit_double_gaussian, do_fitting_double_updated,
    fit_triple_gaussian, do_fitting_triple_updated,
    create_fitting_config,
    create_fitting_config_2d,
    fit_element_edges_2d,
    fit_element_edges_1d,
    fit_multiple_elements_1d,
    fit_multiple_elements_2d,
    print_fitting_summary,
    plot_2d_maps,
    plot_individual_element_map,
    plot_individual_element_maps,
    analyze_spatial_patterns,
    calculate_peak_areas, calculate_peak_areas_fwhm,
    reshape_results_to_2d,
    do_fitting_double_updated_2d,
    fit_double_gaussian_2d
)
from .analysis import (
    calculate_peak_area_ratios, calculate_peak_area_ratio_row,
    calculate_row_gaussian_height_ratio, calculate_gaussian_height_ratio
)

from .plots import (
	plot_area_ratio_with_oxygen_dE_updated,
    plot_element_centers_flexible
)