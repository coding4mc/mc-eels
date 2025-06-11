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


# 1D
# Notebook Code for Line Processing (1D - Quick Check)
# Use this for fast analysis of a single row/line of spectra


def create_fitting_config(element_name, 
                         l3_range, 
                         l2_range=None,
                         l3_peak_range=8, 
                         l2_peak_range=8,
                         spectrum_idx=10,  # Single number for column index
                         show_plots=False,
                         use_triple_fit=False,
                         triple_range=None,
                         triple_peak_range=20):
    """
    Create fitting configuration for 1D line processing.
    
    Parameters:
    ----------
    element_name : str
        Name of the element
    l3_range : list
        [min, max] energy range for L3 edge fitting
    l2_range : list, optional
        [min, max] energy range for L2 edge fitting
    l3_peak_range : float
        Range around peak center for L3 fitting
    l2_peak_range : float
        Range around peak center for L2 fitting
    spectrum_idx : int
        Column index for diagnostic plots (processes row 0)
    show_plots : bool
        Whether to show diagnostic plots
    use_triple_fit : bool
        Whether to use triple Gaussian fitting
    triple_range : list, optional
        [min, max] energy range for triple Gaussian fitting
    triple_peak_range : float
        Range around peak center for triple fitting
    """
    config = {
        'element_name': element_name,
        'l3_range': l3_range,
        'l3_peak_range': l3_peak_range,
        'spectrum_idx': spectrum_idx,
        'show_plots': show_plots,
        'use_triple_fit': use_triple_fit
    }
    
    if l2_range is not None:
        config['l2_range'] = l2_range
        config['l2_peak_range'] = l2_peak_range
    
    if use_triple_fit and triple_range is not None:
        config['triple_range'] = triple_range
        config['triple_peak_range'] = triple_peak_range
    
    return config

def fit_element_edges_1d(element_signal, fitting_config):
    """
    Fit Gaussian peaks for an element - 1D line processing (row 0 only).
    
    Parameters:
    ----------
    element_signal : HyperSpy signal
        Background-removed element signal
    fitting_config : dict
        Fitting configuration from create_fitting_config()
        
    Returns:
    -------
    dict
        Dictionary containing fitting results for the first row
    """
    element_name = fitting_config['element_name']
    spectrum_idx = fitting_config['spectrum_idx']
    show_plots = fitting_config['show_plots']
    
    results = {
        'element_name': element_name,
        'fitting_config': fitting_config
    }
    
    print(f"\nProcessing {element_name} (1D - Row 0 only)...")
    
    # Fit L3 edge
    print(f"Fitting {element_name} L3 edge")
    l3_results = do_fitting_double_updated(
        element_signal, 
        spectrum_idx, 
        fitting_config['l3_range'], 
        peak_range=fitting_config['l3_peak_range'],
        show_plots=show_plots
    )
    
    # Store L3 results
    results['l3_g1_coeffs'] = l3_results[0]
    results['l3_g2_coeffs'] = l3_results[1] 
    results['l3_combined_coeffs'] = l3_results[2]
    
    # Calculate L3 areas and centers using Simpson's rule
    l3_area, l3_center = calculate_peak_areas(
        element_name, 'L3', 
        results['l3_combined_coeffs'], 
        energy_range=fitting_config['l3_range']
    )
    
    results['l3_area'] = l3_area
    results['l3_center'] = l3_center
    
    # Calculate L3 maximum intensities
    x_range = np.linspace(fitting_config['l3_range'][0] - 10, 
                         fitting_config['l3_range'][1] + 10, 10000)
    l3_max_intensities = np.array([np.max(double_gauss(x_range, *coeff)) 
                                  for coeff in results['l3_combined_coeffs']])
    results['l3_max_intensities'] = l3_max_intensities
    
    # Fit L2 edge if specified
    if 'l2_range' in fitting_config:
        print(f"Fitting {element_name} L2 edge")
        l2_results = do_fitting_double_updated(
            element_signal, 
            spectrum_idx, 
            fitting_config['l2_range'], 
            peak_range=fitting_config['l2_peak_range'],
            show_plots=show_plots
        )
        
        # Store L2 results
        results['l2_g1_coeffs'] = l2_results[0]
        results['l2_g2_coeffs'] = l2_results[1]
        results['l2_combined_coeffs'] = l2_results[2]
        
        # Calculate L2 areas and centers using Simpson's rule
        l2_area, l2_center = calculate_peak_areas(
            element_name, 'L2', 
            results['l2_combined_coeffs'], 
            energy_range=fitting_config['l2_range']
        )
        
        results['l2_area'] = l2_area
        results['l2_center'] = l2_center
        
        # Calculate L2 maximum intensities
        x_range_l2 = np.linspace(fitting_config['l2_range'][0] - 10, 
                                fitting_config['l2_range'][1] + 10, 10000)
        l2_max_intensities = np.array([np.max(double_gauss(x_range_l2, *coeff)) 
                                      for coeff in results['l2_combined_coeffs']])
        results['l2_max_intensities'] = l2_max_intensities
        
        # Calculate L3/L2 ratios using Simpson's rule
        results['area_ratio'] = l3_area / l2_area
        results['intensity_ratio'] = l3_max_intensities / l2_max_intensities
        
        print(f"✓ {element_name} L3/L2 fitting completed (1D)")
    else:
        print(f"✓ {element_name} L3 fitting completed (1D)")
    
    # Fit triple Gaussian if specified
    if fitting_config.get('use_triple_fit', False) and 'triple_range' in fitting_config:
        print(f"Fitting {element_name} with triple Gaussian")
        triple_results = do_fitting_triple_updated(
            element_signal,
            spectrum_idx,
            fitting_config['triple_range'],
            peak_range=fitting_config['triple_peak_range'],
            show_plots=show_plots
        )
        
        results['triple_g1_coeffs'] = triple_results[0]
        results['triple_g2_coeffs'] = triple_results[1]
        results['triple_g3_coeffs'] = triple_results[2]
        results['triple_combined_coeffs'] = triple_results[3]
        
        # Calculate triple fit centers
        x_range_triple = np.linspace(fitting_config['triple_range'][0] - 10,
                                   fitting_config['triple_range'][1] + 10, 10000)
        triple_centers = np.array([x_range_triple[np.argmax(triple_gauss(x_range_triple, *coeff))] 
                                 for coeff in results['triple_combined_coeffs']])
        results['triple_centers'] = triple_centers
        
        print(f"✓ {element_name} triple Gaussian fitting completed (1D)")
    
    return results

def fit_multiple_elements_1d(element_signals, fitting_configs):
    """
    Fit Gaussian peaks for multiple elements - 1D processing.
    
    Parameters:
    ----------
    element_signals : dict
        Dictionary of element signals (from background removal)
    fitting_configs : dict
        Dictionary of fitting configurations for each element
        
    Returns:
    -------
    dict
        Dictionary containing fitting results for all elements
    """
    all_results = {}
    
    for element_name, config in fitting_configs.items():
        if element_name not in element_signals:
            print(f"Warning: {element_name} signal not found. Skipping.")
            continue
        
        try:
            element_results = fit_element_edges_1d(element_signals[element_name], config)
            all_results[element_name] = element_results
        except Exception as e:
            print(f"Error fitting {element_name}: {str(e)}")
            continue
    
    return all_results

def print_fitting_summary(fitting_results):
    """
    Print a summary of fitting results.
    
    Parameters:
    ----------
    fitting_results : dict
        Results from fit_element_edges() or fit_multiple_elements()
    """
    print("\n" + "="*60)
    print("GAUSSIAN FITTING SUMMARY")
    print("="*60)
    
    for element_name, results in fitting_results.items():
        print(f"\n{element_name}:")
        print(f"  L3 area (Simpson): {results['l3_area'].mean():.2f} ± {results['l3_area'].std():.2f}")
        
        if 'l2_area' in results:
            print(f"  L2 area (Simpson): {results['l2_area'].mean():.2f} ± {results['l2_area'].std():.2f}")
            print(f"  L3/L2 ratio (Simpson): {results['area_ratio'].mean():.3f} ± {results['area_ratio'].std():.3f}")
            print(f"  L3/L2 ratio (intensity): {results['intensity_ratio'].mean():.3f} ± {results['intensity_ratio'].std():.3f}")


# 2D Array

# Notebook Code for 2D Processing (Full Dataset)
# Use this for complete analysis of all spectra in your 2D dataset

def create_fitting_config_2d(element_name, 
                            l3_range, 
                            l2_range=None,
                            l3_peak_range=8, 
                            l2_peak_range=8,
                            spectrum_idx=(10, 20),  # Tuple for (row, col)
                            show_plots=False,
                            use_triple_fit=False,
                            triple_range=None,
                            triple_peak_range=20):
    """
    Create fitting configuration for 2D processing.
    
    Parameters:
    ----------
    element_name : str
        Name of the element
    l3_range : list
        [min, max] energy range for L3 edge fitting
    l2_range : list, optional
        [min, max] energy range for L2 edge fitting
    l3_peak_range : float
        Range around peak center for L3 fitting
    l2_peak_range : float
        Range around peak center for L2 fitting
    spectrum_idx : tuple
        (row, col) indices for diagnostic plots
    show_plots : bool
        Whether to show diagnostic plots
    use_triple_fit : bool
        Whether to use triple Gaussian fitting
    triple_range : list, optional
        [min, max] energy range for triple Gaussian fitting
    triple_peak_range : float
        Range around peak center for triple fitting
    """
    config = {
        'element_name': element_name,
        'l3_range': l3_range,
        'l3_peak_range': l3_peak_range,
        'spectrum_idx': spectrum_idx,
        'show_plots': show_plots,
        'use_triple_fit': use_triple_fit
    }
    
    if l2_range is not None:
        config['l2_range'] = l2_range
        config['l2_peak_range'] = l2_peak_range
    
    if use_triple_fit and triple_range is not None:
        config['triple_range'] = triple_range
        config['triple_peak_range'] = triple_peak_range
    
    return config

def fit_element_edges_2d(element_signal, fitting_config):
    """
    Fit Gaussian peaks for an element - 2D processing (ALL spectra).
    
    Parameters:
    ----------
    element_signal : HyperSpy signal
        Background-removed element signal
    fitting_config : dict
        Fitting configuration from create_fitting_config_2d()
        
    Returns:
    -------
    dict
        Dictionary containing fitting results for ALL spectra
    """
    element_name = fitting_config['element_name']
    spectrum_idx = fitting_config['spectrum_idx']
    show_plots = fitting_config['show_plots']
    
    results = {
        'element_name': element_name,
        'fitting_config': fitting_config,
        'signal_shape': element_signal.data.shape
    }
    
    print(f"\nProcessing {element_name} (2D - ALL spectra)...")
    
    # Fit L3 edge for ALL spectra
    print(f"Fitting {element_name} L3 edge (2D)")
    l3_results = do_fitting_double_updated_2d(
        element_signal, 
        spectrum_idx, 
        fitting_config['l3_range'], 
        peak_range=fitting_config['l3_peak_range'],
        show_plots=show_plots
    )
    
    # Store L3 results
    results['l3_g1_coeffs'] = l3_results[0]
    results['l3_g2_coeffs'] = l3_results[1] 
    results['l3_combined_coeffs'] = l3_results[2]
    
    # Calculate L3 areas and centers using Simpson's rule
    l3_area, l3_center = calculate_peak_areas(
        element_name, 'L3', 
        results['l3_combined_coeffs'], 
        energy_range=fitting_config['l3_range']
    )
    
    results['l3_area'] = l3_area
    results['l3_center'] = l3_center
    
    # Calculate L3 maximum intensities
    x_range = np.linspace(fitting_config['l3_range'][0] - 10, 
                         fitting_config['l3_range'][1] + 10, 10000)
    l3_max_intensities = np.array([np.max(double_gauss(x_range, *coeff)) 
                                  for coeff in results['l3_combined_coeffs']])
    results['l3_max_intensities'] = l3_max_intensities
    
    # Fit L2 edge if specified
    if 'l2_range' in fitting_config:
        print(f"Fitting {element_name} L2 edge (2D)")
        l2_results = do_fitting_double_updated_2d(
            element_signal, 
            spectrum_idx, 
            fitting_config['l2_range'], 
            peak_range=fitting_config['l2_peak_range'],
            show_plots=show_plots
        )
        
        # Store L2 results
        results['l2_g1_coeffs'] = l2_results[0]
        results['l2_g2_coeffs'] = l2_results[1]
        results['l2_combined_coeffs'] = l2_results[2]
        
        # Calculate L2 areas and centers using Simpson's rule
        l2_area, l2_center = calculate_peak_areas(
            element_name, 'L2', 
            results['l2_combined_coeffs'], 
            energy_range=fitting_config['l2_range']
        )
        
        results['l2_area'] = l2_area
        results['l2_center'] = l2_center
        
        # Calculate L2 maximum intensities
        x_range_l2 = np.linspace(fitting_config['l2_range'][0] - 10, 
                                fitting_config['l2_range'][1] + 10, 10000)
        l2_max_intensities = np.array([np.max(double_gauss(x_range_l2, *coeff)) 
                                      for coeff in results['l2_combined_coeffs']])
        results['l2_max_intensities'] = l2_max_intensities
        
        # Calculate L3/L2 ratios using Simpson's rule
        results['area_ratio'] = l3_area / l2_area
        results['intensity_ratio'] = l3_max_intensities / l2_max_intensities
        
        print(f"✓ {element_name} L3/L2 fitting completed (2D)")
    else:
        print(f"✓ {element_name} L3 fitting completed (2D)")
    
    # Fit triple Gaussian if specified
    if fitting_config.get('use_triple_fit', False) and 'triple_range' in fitting_config:
        print(f"Fitting {element_name} with triple Gaussian (2D)")
        triple_results = do_fitting_triple_updated_2d(
            element_signal,
            spectrum_idx,
            fitting_config['triple_range'],
            peak_range=fitting_config['triple_peak_range'],
            show_plots=show_plots
        )
        
        results['triple_g1_coeffs'] = triple_results[0]
        results['triple_g2_coeffs'] = triple_results[1]
        results['triple_g3_coeffs'] = triple_results[2]
        results['triple_combined_coeffs'] = triple_results[3]
        
        # Calculate triple fit centers
        x_range_triple = np.linspace(fitting_config['triple_range'][0] - 10,
                                   fitting_config['triple_range'][1] + 10, 10000)
        triple_centers = np.array([x_range_triple[np.argmax(triple_gauss(x_range_triple, *coeff))] 
                                 for coeff in results['triple_combined_coeffs']])
        results['triple_centers'] = triple_centers
        
        print(f"✓ {element_name} triple Gaussian fitting completed (2D)")
    
    return results

def fit_multiple_elements_2d(element_signals, fitting_configs):
    """
    Fit Gaussian peaks for multiple elements - 2D processing.
    
    Parameters:
    ----------
    element_signals : dict
        Dictionary of element signals (from background removal)
    fitting_configs : dict
        Dictionary of fitting configurations for each element
        
    Returns:
    -------
    dict
        Dictionary containing fitting results for all elements
    """
    all_results = {}
    
    for element_name, config in fitting_configs.items():
        if element_name not in element_signals:
            print(f"Warning: {element_name} signal not found. Skipping.")
            continue
        
        try:
            element_results = fit_element_edges_2d(element_signals[element_name], config)
            all_results[element_name] = element_results
        except Exception as e:
            print(f"Error fitting {element_name}: {str(e)}")
            continue
    
    print(f"\n✅ 2D fitting completed for all elements!")
    return all_results

def plot_2d_results(fitting_results_2d, element_name, result_type='area_ratio'):
    """
    Plot 2D maps of fitting results.
    
    Parameters:
    ----------
    fitting_results_2d : dict
        Results from fit_multiple_elements_2d()
    element_name : str
        Element to plot
    result_type : str
        Type of result to plot ('area_ratio', 'l3_center', 'l2_center', etc.)
    """
    if element_name not in fitting_results_2d:
        print(f"Element {element_name} not found in results.")
        return
    
    if result_type not in fitting_results_2d[element_name]:
        print(f"Result type {result_type} not found for {element_name}.")
        return
    
    # Get the data and reshape to 2D
    data_1d = fitting_results_2d[element_name][result_type]
    signal_shape = fitting_results_2d[element_name]['signal_shape']
    data_2d = reshape_results_to_2d(data_1d, signal_shape)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    im = plt.imshow(data_2d, cmap='viridis', aspect='auto')
    plt.colorbar(im, label=f'{element_name} {result_type}')
    plt.title(f'{element_name} {result_type} Map')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def plot_2d_maps(fitting_results_2d, elements=['Ni', 'Mn', 'Co'], include_oxygen=True, 
                 figsize=(12, 8), cmap_elements='viridis', cmap_oxygen='plasma'):
    """
    Create 2D visualization maps for element L3/L2 ratios and oxygen ΔE.
    
    Parameters:
    ----------
    fitting_results_2d : dict
        Results from fit_multiple_elements_2d()
    elements : list
        List of elements to plot (default: ['Ni', 'Mn', 'Co'])
    include_oxygen : bool
        Whether to include oxygen ΔE plot (default: True)
    figsize : tuple
        Figure size (width, height) (default: (12, 8))
    cmap_elements : str
        Colormap for element ratio plots (default: 'viridis')
    cmap_oxygen : str
        Colormap for oxygen ΔE plot (default: 'plasma')
        
    Returns:
    -------
    fig, axes : matplotlib objects
        Figure and axes objects for further customization
    """
    if not fitting_results_2d:
        print("❌ No fitting results provided")
        return None, None
    
    print("Creating 2D maps...")
    
    # Count available plots
    available_elements = []
    for element in elements:
        if element in fitting_results_2d and 'area_ratio' in fitting_results_2d[element]:
            available_elements.append(element)
    
    # Check for oxygen
    has_oxygen = (include_oxygen and 
                  'O_prepeak' in fitting_results_2d and 
                  'O_main' in fitting_results_2d)
    
    total_plots = len(available_elements) + (1 if has_oxygen else 0)
    
    if total_plots == 0:
        print("❌ No valid data to plot")
        return None, None
    
    print(f"✅ Will plot {len(available_elements)} elements" + 
          (f" + oxygen ΔE" if has_oxygen else ""))
    
    # Determine subplot layout
    if total_plots <= 2:
        rows, cols = 1, total_plots
        figsize = (figsize[0] // 2 * total_plots, figsize[1] // 2)
    elif total_plots <= 4:
        rows, cols = 2, 2
    elif total_plots <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if total_plots == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot element ratios
    for element in available_elements:
        try:
            data_1d = fitting_results_2d[element]['area_ratio']
            signal_shape = fitting_results_2d[element]['signal_shape']
            data_2d = reshape_results_to_2d(data_1d, signal_shape)
            
            im = axes[plot_idx].imshow(data_2d, cmap=cmap_elements, aspect='auto')
            axes[plot_idx].set_title(f'{element} L3/L2 Ratio')
            axes[plot_idx].set_xlabel('Column Index (0-63)')
            axes[plot_idx].set_ylabel('Row Index (0-5)')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[plot_idx])
            
            # Add statistics as text
            stats_text = (f'Mean: {data_1d.mean():.3f}\n'
                         f'Std: {data_1d.std():.3f}\n'
                         f'Range: {data_1d.min():.3f}-{data_1d.max():.3f}')
            
            axes[plot_idx].text(0.02, 0.98, stats_text, 
                               transform=axes[plot_idx].transAxes,
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               fontsize=8)
            
            plot_idx += 1
            print(f"  ✅ Plotted {element}")
            
        except Exception as e:
            print(f"  ❌ Error plotting {element}: {str(e)}")
            continue
    
    # Plot oxygen delta E if available
    if has_oxygen:
        try:
            o_prepeak_centers = fitting_results_2d['O_prepeak']['l3_center']
            o_main_centers = fitting_results_2d['O_main']['triple_centers']
            o_delta_e_1d = o_main_centers - o_prepeak_centers
            
            signal_shape = fitting_results_2d['O_prepeak']['signal_shape']
            o_delta_e_2d = reshape_results_to_2d(o_delta_e_1d, signal_shape)
            
            im = axes[plot_idx].imshow(o_delta_e_2d, cmap=cmap_oxygen, aspect='auto')
            axes[plot_idx].set_title('Oxygen ΔE')
            axes[plot_idx].set_xlabel('Column Index (0-63)')
            axes[plot_idx].set_ylabel('Row Index (0-5)')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[plot_idx], label='ΔE (eV)')
            
            # Add statistics as text
            stats_text = (f'Mean: {o_delta_e_1d.mean():.2f} eV\n'
                         f'Std: {o_delta_e_1d.std():.2f} eV\n'
                         f'Range: {o_delta_e_1d.min():.2f}-{o_delta_e_1d.max():.2f} eV')
            
            axes[plot_idx].text(0.02, 0.98, stats_text, 
                               transform=axes[plot_idx].transAxes,
                               verticalalignment='top', 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               fontsize=8)
            
            plot_idx += 1
            print(f"  ✅ Plotted oxygen ΔE")
            
        except Exception as e:
            print(f"  ❌ Error plotting oxygen: {str(e)}")
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Successfully plotted {plot_idx} maps")
    
    return fig, axes


# Alternative function for individual element plots
def plot_individual_element_map(fitting_results_2d, element, figsize=(10, 3), 
                               cmap='viridis', save_path=None):
    """
    Plot an individual element map.
    
    Parameters:
    ----------
    fitting_results_2d : dict
        Results from fit_multiple_elements_2d()
    element : str
        Element name to plot ('Ni', 'Mn', 'Co', etc.)
    figsize : tuple
        Figure size (width, height) (default: (10, 3))
    cmap : str
        Colormap to use (default: 'viridis')
    save_path : str, optional
        Path to save the figure (if None, just display)
        
    Returns:
    -------
    fig, ax : matplotlib objects
        Figure and axis objects
    """
    if element not in fitting_results_2d:
        print(f"❌ Element {element} not found in results")
        return None, None
    
    if 'area_ratio' not in fitting_results_2d[element]:
        print(f"❌ No L3/L2 ratio data for {element}")
        return None, None
    
    # Get data and reshape
    data_1d = fitting_results_2d[element]['area_ratio']
    signal_shape = fitting_results_2d[element]['signal_shape']
    data_2d = reshape_results_to_2d(data_1d, signal_shape)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data_2d, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=f'{element} L3/L2 Ratio')
    
    # Labels and title
    ax.set_title(f'{element} L3/L2 Ratio Map ({data_2d.shape[0]} rows × {data_2d.shape[1]} columns)')
    ax.set_xlabel('Column Index (0-63)')
    ax.set_ylabel('Row Index (0-5)')
    
    # Add comprehensive statistics
    stats_text = (f'Statistics:\n'
                 f'Mean: {data_1d.mean():.3f}\n'
                 f'Median: {np.median(data_1d):.3f}\n'
                 f'Std: {data_1d.std():.3f}\n'
                 f'Min: {data_1d.min():.3f}\n'
                 f'Max: {data_1d.max():.3f}\n'
                 f'Count: {len(data_1d)} spectra')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=9)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot to {save_path}")
    
    plt.show()
    
    print(f"✅ Plotted {element} map: {data_2d.shape}")
    
    return fig, ax