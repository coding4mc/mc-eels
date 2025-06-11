"""
Gaussian fitting functions for EELS data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simpson as simps


def gauss(x, h, sigma, center):
    """
    Gaussian function.
    
    Parameters:
    ----------
    x : numpy.ndarray
        X values
    h : float
        Height of the Gaussian
    sigma : float
        Standard deviation of the Gaussian
    center : float
        Center of the Gaussian
        
    Returns:
    -------
    numpy.ndarray
        Gaussian values
    """
    return h * np.exp(-((x-center)**2) / (2*sigma**2))


def double_gauss(x, a0, sigma0, center0, a1, sigma1, center1):
    """
    Sum of two Gaussians.
    
    Parameters:
    ----------
    x : numpy.ndarray
        X values
    a0, sigma0, center0 : float
        Parameters for first Gaussian
    a1, sigma1, center1 : float
        Parameters for second Gaussian
        
    Returns:
    -------
    numpy.ndarray
        Sum of two Gaussians
    """
    return gauss(x, a0, sigma0, center0) + gauss(x, a1, sigma1, center1)


def triple_gauss(x, a0, sigma0, center0, a1, sigma1, center1, a2, sigma2, center2):
    """
    Sum of three Gaussians.
    
    Parameters:
    ----------
    x : numpy.ndarray
        X values
    a0, sigma0, center0 : float
        Parameters for first Gaussian
    a1, sigma1, center1 : float
        Parameters for second Gaussian
    a2, sigma2, center2 : float
        Parameters for third Gaussian
        
    Returns:
    -------
    numpy.ndarray
        Sum of three Gaussians
    """
    return gauss(x, a0, sigma0, center0) + gauss(x, a1, sigma1, center1) + gauss(x, a2, sigma2, center2)


def get_signal_x_axis(signal):
    """
    Get x-axis values from a signal.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to get x-axis from
        
    Returns:
    -------
    numpy.ndarray
        X-axis values
    """
    signal_axes = signal.axes_manager.signal_axes[0]
    x_min = signal_axes.offset
    x_max = x_min + signal_axes.scale * signal_axes.size
    return np.linspace(x_min, x_max, signal_axes.size)


def fit_double_gaussian(signal, signal_range, peak_range=1.5, manual_fine_tune=False):
    """
    Fit double Gaussian to a signal.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to fit
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
    manual_fine_tune : bool
        Whether to manually fine-tune the fit
        
    Returns:
    -------
    list
        List of double Gaussian coefficients for each spectrum
    """
    all_x = get_signal_x_axis(signal)
    
    # Build x array within signal range
    signal_range_min, signal_range_max = signal_range
    signal_range_min_index = np.argmin(np.abs(all_x - signal_range_min))
    signal_range_max_index = np.argmin(np.abs(all_x - signal_range_max))
    x_signal_range = all_x[signal_range_min_index : signal_range_max_index]

    coeffs = []
    for index, data_record in enumerate(signal.data[0]):
        y_signal_range = data_record[signal_range_min_index : signal_range_max_index]
        
        x_peak_index = np.argmax(y_signal_range)
        x_peak = x_signal_range[x_peak_index]
        x_min = x_peak - peak_range
        x_max = x_peak + peak_range
        
        # Find an array of X values start from x_min to x_max
        x_min_index = np.argmin(np.abs(x_signal_range - x_min))
        x_max_index = np.argmin(np.abs(x_signal_range - x_max))
        x = x_signal_range[x_min_index : x_max_index]
        y = y_signal_range[x_min_index : x_max_index]

        # Improved fit parameters
        min_height_0 = 0
        max_height_0 = max(y) * 1.2
        min_height_1 = 0
        max_height_1 = max(y) * 1.2
        min_center_0, max_center_0 = x[0], x[-1]
        min_center_1, max_center_1 = x[0], x[-1]
        min_sigma = 0.1
        max_sigma = 3.0

        # Improved bounds
        min_bound = [min_height_0, min_sigma, min_center_0, min_height_1, min_sigma, min_center_1] 
        max_bound = [max_height_0, max_sigma, max_center_0, max_height_1, max_sigma, max_center_1]

        # Better initial guesses
        initial_height_0 = np.mean([min_height_0, max_height_0]) * 0.7
        initial_height_1 = np.mean([min_height_1, max_height_1]) * 0.5
        initial_center_0 = x_peak - 0.5
        initial_center_1 = x_peak + 0.5
        initial_sigma = 1.0
        
        initial_conditions = [initial_height_0, initial_sigma, initial_center_0, 
                             initial_height_1, initial_sigma, initial_center_1]
        max_iterations = 100000
        
        try:
            coeff, _ = curve_fit(double_gauss, x, y, p0=initial_conditions, maxfev=max_iterations, 
                                bounds=(min_bound, max_bound))
            
            # Sort by gaussian centers
            c0, c1 = coeff[2], coeff[5]
            if c0 > c1:
                coeff = [*coeff[3:], *coeff[:3]]
                
        except Exception as e:
            print(f"Fitting failed for index {index}: {str(e)}")
            # Use previous coefficients if available, otherwise use initial guess
            if index > 0 and coeffs:
                coeff = coeffs[index - 1]
            else:
                coeff = initial_conditions
                
        coeffs.append(coeff)
    
    return coeffs


def do_fitting_double_updated(element, index, signal_range, peak_range=1.5, manual_fine_tune=False, show_plots=True):
    """
    Fit double Gaussian with updated labels (g1 and g2 instead of K1a and K1b).
    
    Parameters:
    ----------
    element : HyperSpy signal
        Element signal to fit
    index : int
        Index of spectrum to process and plot
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
    manual_fine_tune : bool
        Whether to manually fine-tune the fit
    show_plots : bool
        Whether to show diagnostic plots
        
    Returns:
    -------
    g1_coeffs, g2_coeffs, g12_coeffs, g1_y, g2_y, g12_y
    """
    g12_coeffs = fit_double_gaussian(element, signal_range=signal_range, peak_range=peak_range, 
                                    manual_fine_tune=manual_fine_tune)
    g1_coeffs = [coeff[:3] for coeff in g12_coeffs]
    g2_coeffs = [coeff[3:] for coeff in g12_coeffs]
    
    g12_coeff = g12_coeffs[index]
    g1_coeff = g12_coeff[:3]
    g2_coeff = g12_coeff[3:]

    x = get_signal_x_axis(element)
    g1_y = gauss(x, *g1_coeff)
    g2_y = gauss(x, *g2_coeff)
    g12_y = double_gauss(x, *g12_coeff)

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(x, element.data[0][index], 'rx', label="Raw data")
        plt.plot(x, g1_y, label="g1")
        plt.plot(x, g2_y, label="g2")
        plt.plot(x, g12_y, label="g1 + g2")
        plt.legend()
        plt.xlabel("eV")
        plt.ylabel("Intensity")
        plt.tight_layout()
        plt.show()
    
    return g1_coeffs, g2_coeffs, g12_coeffs, g1_y, g2_y, g12_y



def calculate_peak_areas(element, edge_type, g12_coeffs, energy_range=[600, 700], num_points=10000):
    """
    Calculate the area under fitted Gaussian peaks for L3 or L2 edges.
    
    Parameters:
    ----------
    element : str
        Element name ('Mn', 'Co', or 'Ni')
    edge_type : str
        Edge type ('L3' or 'L2')
    g12_coeffs : list
        List of double Gaussian coefficients for each spectrum
    energy_range : list
        Range of energies to use for integration [min, max]
    num_points : int
        Number of points to use for integration
        
    Returns:
    -------
    areas : numpy array
        Array of integrated areas for each spectrum
    centers : numpy array
        Array of peak centers for each spectrum
    """
    # Define energy range for integration
    x_coeff = np.linspace(energy_range[0], energy_range[1], num_points)
    
    # Calculate peak centers (position of maximum intensity)
    centers = np.array([x_coeff[np.argmax(double_gauss(x_coeff, *coeff))] for coeff in g12_coeffs])
    
    # Calculate areas under the curve using Simpson's rule
    areas = []
    for coeff in g12_coeffs:
        y_values = double_gauss(x_coeff, *coeff)
        area = simps(y_values, x_coeff)
        areas.append(area)
    
    # Convert to numpy array
    areas = np.array(areas)
    
    print(f"{element} {edge_type} peak areas calculated. Range: {np.min(areas):.2f} to {np.max(areas):.2f}")
    
    return areas, centers


def fit_triple_gaussian(signal, signal_range, peak_range=3.0, manual_fine_tune=False):
    """
    Fit triple Gaussian to a signal, specifically designed for oxygen main peak.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to fit
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
    manual_fine_tune : bool
        Whether to manually fine-tune the fit
        
    Returns:
    -------
    coeffs : list
        List of triple Gaussian coefficients for each spectrum
    """
    all_x = get_signal_x_axis(signal)
    
    # Build x array within signal range
    signal_range_min, signal_range_max = signal_range
    signal_range_min_index = np.argmin(np.abs(all_x - signal_range_min))
    signal_range_max_index = np.argmin(np.abs(all_x - signal_range_max))
    x_signal_range = all_x[signal_range_min_index : signal_range_max_index]

    coeffs = []
    for index, data_record in enumerate(signal.data[0]):
        y_signal_range = data_record[signal_range_min_index : signal_range_max_index]
        
        x_peak_index = np.argmax(y_signal_range)
        x_peak = x_signal_range[x_peak_index]
        x_min = x_peak - peak_range
        x_max = x_peak + peak_range
        
        # Find an array of X values start from x_min to x_max
        x_min_index = np.argmin(np.abs(x_signal_range - x_min))
        x_max_index = np.argmin(np.abs(x_signal_range - x_max))
        x = x_signal_range[x_min_index : x_max_index]
        y = y_signal_range[x_min_index : x_max_index]

        # Set parameters for fitting
        max_height = max(y) * 1.2
        min_center, max_center = x[0], x[-1]
        min_sigma = 0.1
        max_sigma = 3.0

        # Parameters for triple Gaussian
        min_bound = [0, min_sigma, min_center, 0, min_sigma, min_center, 0, min_sigma, min_center] 
        max_bound = [max_height, max_sigma, max_center, max_height, max_sigma, max_center, max_height, max_sigma, max_center]

        # Initial guesses - distribute centers across the peak range
        peak_width = max_center - min_center
        initial_height = max_height * 0.4
        initial_center_0 = min_center + peak_width * 0.2
        initial_center_1 = min_center + peak_width * 0.5  # Middle
        initial_center_2 = min_center + peak_width * 0.8
        initial_sigma = peak_width / 6  # Reasonable starting width
        
        initial_conditions = [
            initial_height, initial_sigma, initial_center_0,
            initial_height, initial_sigma, initial_center_1, 
            initial_height, initial_sigma, initial_center_2
        ]
        max_iterations = 100000
        
        try:
            coeff, _ = curve_fit(triple_gauss, x, y, p0=initial_conditions, maxfev=max_iterations, 
                                bounds=(min_bound, max_bound))
            
            # Sort by gaussian centers
            centers = [coeff[2], coeff[5], coeff[8]]
            idx_sorted = np.argsort(centers)
            
            # Rearrange coefficients based on sorted centers
            sorted_coeff = np.zeros_like(coeff)
            for i, idx in enumerate(idx_sorted):
                sorted_coeff[i*3:(i+1)*3] = coeff[idx*3:(idx+1)*3]
            
            coeff = sorted_coeff
                
        except Exception as e:
            print(f"Triple Gaussian fitting failed for index {index}: {str(e)}")
            # Use previous coefficients if available, otherwise use initial guess
            if index > 0 and coeffs:
                coeff = coeffs[index - 1]
            else:
                coeff = initial_conditions
                
        coeffs.append(coeff)
    
    return coeffs


def do_fitting_triple_updated(element, index, signal_range, peak_range=3.0, manual_fine_tune=False, show_plots=True):
    """
    Fit triple Gaussian for oxygen main peak.
    
    Parameters:
    ----------
    element : HyperSpy signal
        Element signal to fit
    index : int
        Index of spectrum to process and plot
    signal_range : list
        Range of energies to fit [min, max]
    peak_range : float
        Range around peak center to use for fitting
    manual_fine_tune : bool
        Whether to manually fine-tune the fit
    show_plots : bool
        Whether to show diagnostic plots
        
    Returns:
    -------
    g1_coeffs, g2_coeffs, g3_coeffs, g123_coeffs, g123_y
    """
    g123_coeffs = fit_triple_gaussian(element, signal_range=signal_range, peak_range=peak_range, 
                                    manual_fine_tune=manual_fine_tune)
    
    # Extract individual Gaussian coefficients
    g1_coeffs = [coeff[0:3] for coeff in g123_coeffs]
    g2_coeffs = [coeff[3:6] for coeff in g123_coeffs]
    g3_coeffs = [coeff[6:9] for coeff in g123_coeffs]
    
    # Get coefficients for the requested index
    g123_coeff = g123_coeffs[index]
    g1_coeff = g123_coeff[0:3]
    g2_coeff = g123_coeff[3:6]
    g3_coeff = g123_coeff[6:9]

    # Get x-axis and calculate fitted curves
    x = get_signal_x_axis(element)
    g1_y = gauss(x, *g1_coeff)
    g2_y = gauss(x, *g2_coeff)
    g3_y = gauss(x, *g3_coeff)
    g123_y = triple_gauss(x, *g123_coeff)

    # Show diagnostic plot if requested
    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(x, element.data[0][index], 'rx', label="Raw data")
        plt.plot(x, g1_y, label="g1", color='#1f77b4')  # Blue
        plt.plot(x, g2_y, label="g2", color='#ff7f0e')  # Orange
        plt.plot(x, g3_y, label="g3", color='#2ca02c')  # Green
        plt.plot(x, g123_y, label="g1 + g2 + g3", color='red')
        plt.legend()
        plt.xlabel("eV")
        plt.ylabel("Intensity")
        plt.title(f"Triple Gaussian Fit - O Main Peak")
        plt.tight_layout()
        plt.show()
    
    return g1_coeffs, g2_coeffs, g3_coeffs, g123_coeffs, g123_y



def calculate_peak_areas_fwhm(element, edge_type, g12_coeffs, energy_range=[600, 700], num_points=10000):
    """
    Calculate the area under fitted Gaussian peaks using FWHM method for L3 or L2 edges.
    
    Parameters:
    ----------
    element : str
        Element name ('Mn', 'Co', or 'Ni')
    edge_type : str
        Edge type ('L3' or 'L2')
    g12_coeffs : list
        List of double Gaussian coefficients for each spectrum
    energy_range : list
        Range of energies to use for integration [min, max]
    num_points : int
        Number of points to use for integration
        
    Returns:
    -------
    areas : numpy array
        Array of integrated areas for each spectrum
    centers : numpy array
        Array of peak centers for each spectrum
    fwhm_values : numpy array
        Array of FWHM values for each spectrum
    """
    # Define energy range for integration
    x_coeff = np.linspace(energy_range[0], energy_range[1], num_points)
    
    # Calculate peak centers (position of maximum intensity)
    centers = []
    areas = []
    fwhm_values = []
    
    for coeff in g12_coeffs:
        # Calculate peak curve
        y_values = double_gauss(x_coeff, *coeff)
        
        # Find maximum peak position and value
        max_idx = np.argmax(y_values)
        max_pos = x_coeff[max_idx]
        max_val = y_values[max_idx]
        centers.append(max_pos)
        
        # Calculate FWHM for each Gaussian component
        # Formula for Gaussian FWHM = 2 * sigma * sqrt(2 * ln(2)) ≈ 2.355 * sigma
        fwhm_g1 = 2.355 * coeff[1]  # FWHM of first Gaussian
        fwhm_g2 = 2.355 * coeff[4]  # FWHM of second Gaussian
        
        # Use weighted average based on peak heights
        weight_g1 = coeff[0] / (coeff[0] + coeff[3])
        weight_g2 = coeff[3] / (coeff[0] + coeff[3])
        fwhm_avg = (fwhm_g1 * weight_g1 + fwhm_g2 * weight_g2)
        fwhm_values.append(fwhm_avg)

        # Get x-axis within FWHM range
        half_fwhm = fwhm_avg / 2
        fwhm_start = max_pos - half_fwhm
        fwhm_end = max_pos + half_fwhm

        # Find indices within FWHM range
        fwhm_mask = (x_coeff >= fwhm_start) & (x_coeff <= fwhm_end)
        x_fwhm = x_coeff[fwhm_mask]
        y_fwhm = y_values[fwhm_mask]

        # Calculate area within FWHM using Simpson's rule
        if len(x_fwhm) > 2:  # Need at least 3 points for Simpson's rule
            area = simps(y_fwhm, x_fwhm)
        else:
            raise ValueError("Cannot calculate area. Only 1 data point within FWHM range.")
        
        areas.append(area)
    
    # Convert to numpy arrays
    areas = np.array(areas)
    centers = np.array(centers)
    fwhm_values = np.array(fwhm_values)
    
    print(f"{element} {edge_type} FWHM peak areas calculated. Range: {np.min(areas):.2f} to {np.max(areas):.2f}")
    print(f"{element} {edge_type} FWHM values. Range: {np.min(fwhm_values):.2f} to {np.max(fwhm_values):.2f} eV")
    
    return areas, centers, fwhm_values
