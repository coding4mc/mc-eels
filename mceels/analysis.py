"""
Analysis functions for EELS data.
"""

import numpy as np
from .gaussian import get_signal_x_axis, fit_double_gaussian, double_gauss


def calculate_peak_area_ratio_row(spectrum_row, signal_range, peak_range, num_points=10000):
    """
    Calculate the ratio of peak areas for a row of spectra.
    
    Parameters:
    ----------
    spectrum_row : HyperSpy signal
        Row of spectra to analyze
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
    num_points : int
        Number of points to use for integration
        
    Returns:
    -------
    numpy.ndarray
        Array of peak area ratios
    """
    from .gaussian import calculate_single_peak_area
    
    g12_coeffs = fit_double_gaussian(spectrum_row, signal_range=signal_range, peak_range=peak_range)

    g1_coeffs = [coeffs[:3] for coeffs in g12_coeffs]
    g1_fwhm_areas, _ = calculate_single_peak_area(
        g1_coeffs, signal_range, num_points
    )

    g2_coeffs = [coeffs[3:] for coeffs in g12_coeffs]
    g2_fwhm_areas, _ = calculate_single_peak_area(
        g2_coeffs, signal_range, num_points
    )

    return g1_fwhm_areas / g2_fwhm_areas


def calculate_peak_area_ratios(spectrum, energy_range, num_points=10000):
    """
    Calculate the area under fitted Gaussian peaks using FWHM method for L3 or L2 edges.
    
    Parameters:
    ----------
    spectrum : EELSSpectrum
        The 3D EELS Spectrum
    energy_range : list
        Range of energies to use for integration [min, max]
    num_points : int
        Number of points to use for integration
        
    Returns:
    -------
    areas : numpy array
        Array of integrated areas for each spectrum
    """
    # Get the shape of the first two dimensions (spatial dimensions)
    height, width = spectrum.data.shape[0], spectrum.data.shape[1]
    print(f"Processing data with dimensions: {height} x {width}")

    # Create an empty array to store the ratios
    ratio_matrix = np.zeros((height, width))
    ratio_matrix.fill(np.nan)  # Initialize with NaN to identify pixels where fitting fails

    # Process each pixel
    for i in range(height):
        # Extract the spectrum at this row
        row_spectrum = spectrum.inav[:, i:i+1]
        
        # Calculate the ratio
        ratio = calculate_peak_area_ratio_row(row_spectrum, energy_range, num_points)
        
        # Store in the ratio array
        ratio_matrix[i] = ratio

    # Normalise the ratios
    ratio_matrix -= np.min(ratio_matrix)
    ratio_matrix /= np.max(ratio_matrix)
    return ratio_matrix


def calculate_row_gaussian_height_ratio(spectrum_row, signal_range, peak_range):
    """
    Calculates the ratio of double Gaussian peaks.

    Parameters:
    ----------
    spectrum_row : EELSSpectrum
        A 2D spectrum (width, spectra)
    signal_range : list
        The x-range of spectra to fit [min, max]
    peak_range : float
        The x-range around the Gaussian peak to fit

    Returns:
    -------
    numpy.ndarray
        1D (width) vector of g1 / g2
    """
    g12_coeffs = fit_double_gaussian(spectrum_row, signal_range=signal_range, peak_range=peak_range)
    ratios = []
    for spectrum_coeffs in g12_coeffs:
        # Extract coefficients for g1 and g2
        g1_coeff = spectrum_coeffs[:3]  # First 3 coefficients are for g1
        g2_coeff = spectrum_coeffs[3:]  # Last 3 coefficients are for g2
        
        # Get heights (a0 parameter)
        g1_height = g1_coeff[0]
        g2_height = g2_coeff[0]
        
        # Calculate ratio
        if g2_height > 0:
            ratio = g1_height / g2_height
        else:
            ratio = 0
            
        ratios.append(ratio)
    return np.array(ratios)


def calculate_gaussian_height_ratio(spectrum, signal_range, peak_range):
    """
    Calculates the ratio of double Gaussian peaks.
    
    Parameters:
    ----------
    spectrum : EELSSpectrum
        A 3D spectrum (height, width, spectra)
    signal_range : list
        The x-range of spectra to fit [min, max]
    peak_range : float
        The x-range around the Gaussian peak to fit

    Returns:
    -------
    numpy.ndarray
        2D (height, width) matrix of g1 / g2
    """
    # Get the shape of the first two dimensions (spatial dimensions)
    height, width = spectrum.data.shape[0], spectrum.data.shape[1]
    print(f"Processing data with dimensions: {height} x {width}")

    # Create an empty array to store the ratios
    ratio_matrix = np.zeros((height, width))
    ratio_matrix.fill(np.nan)  # Initialize with NaN to identify pixels where fitting fails

    # Process each pixel
    for i in range(height):
        # Extract the spectrum at this row
        row_spectrum = spectrum.inav[:, i:i+1]
        
        # Calculate the ratio
        ratio = calculate_row_gaussian_height_ratio(row_spectrum, signal_range, peak_range)
        
        # Store in the ratio array
        ratio_matrix[i] = ratio

    # Normalise the ratios
    ratio_matrix -= np.min(ratio_matrix)
    ratio_matrix /= np.max(ratio_matrix)
    return ratio_matrix
