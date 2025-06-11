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
from .background import process_nmc811_elements, improved_background_removal
from .gaussian import (
    gauss, double_gauss, triple_gauss,
    fit_double_gaussian, do_fitting_double_updated,
    fit_triple_gaussian, do_fitting_triple_updated,
    calculate_peak_areas, calculate_peak_areas_fwhm
)
from .analysis import (
    calculate_peak_area_ratios, calculate_peak_area_ratio_row,
    calculate_row_gaussian_height_ratio, calculate_gaussian_height_ratio
)
from .plotting import (
    color_to_number,
    plot_area_ratio_with_oxygen_dE_updated,
    plot_element_centers_grid,
    save_all_elements_on_one_figure,
    plot_ratio_vs_distance_new, plot_e
)
