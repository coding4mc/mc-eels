"""
Background subtraction functions for EELS data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Default element parameters for common elements
default_element_params = {
    'Ni': {
        'edge_range': [820.0, 900.0],
        'pre_edge_range': [790.0, 850.0],
        'post_edge_range': [885.0, 900.0],
        'edge_region': [850.0, 885.0],
        'poly_order': 3,
        'l3_range': [853.0, 861.0],
        'l2_range': [871.0, 879.0],
        'color': '#00008B',
        'use_two_window': True,
        'force_flat_post_edge': True
    },
    'Co': {
        'edge_range': [720.0, 820.0],
        'pre_edge_range': [720.0, 770.0],
        'post_edge_range': [805, 820.0],
        'edge_region': [775.0, 805.0],
        'poly_order': 2,
        'l3_range': [775, 790.0],
        'l2_range': [792, 800.0],
        'color': '#ff7f00',
        'use_two_window': True,
        'force_flat_post_edge': True
    },
    'Mn': {
        'edge_range': [590.0, 690.0],
        'pre_edge_range': [580.0, 635.0],
        'post_edge_range': [662.0, 680.0],
        'edge_region': [638.0, 660.0],
        'poly_order': 2,
        'l3_range': [640.0, 646.0],
        'l2_range': [650.0, 659.0],
        'color': '#2ca02c',
        'use_two_window': True,
        'force_flat_post_edge': True
    },
    'O': {
        'edge_range': [450.0, 590.0],
        'pre_edge_range': [460.0, 510.0],
        'post_edge_range': [556.0, 580.0],
        'edge_region': [525.0, 560.0],
        'poly_order': 4,
        'l3_range': [525.0, 534.0],
        'l2_range': [535.5, 553.0],
        'color': '#d62728',
        'use_two_window': True,
        'force_flat_post_edge': False
    },
    'Fe': {
        'edge_range': [690.0, 750.0],
        'pre_edge_range': [680.0, 700.0],
        'post_edge_range': [740.0, 750.0],
        'edge_region': [700.0, 740.0],
        'poly_order': 2,
        'l3_range': [706.0, 712.0],
        'l2_range': [719.0, 725.0],
        'color': '#9467bd',
        'use_two_window': True,
        'force_flat_post_edge': True
    },
    'Ti': {
        'edge_range': [440.0, 490.0],
        'pre_edge_range': [430.0, 450.0],
        'post_edge_range': [480.0, 490.0],
        'edge_region': [450.0, 480.0],
        'poly_order': 2,
        'l3_range': [455.0, 460.0],
        'l2_range': [461.0, 466.0],
        'color': '#8c564b',
        'use_two_window': True,
        'force_flat_post_edge': True
    }
}

def create_element_config(element_name, 
                         edge_range, 
                         pre_edge_range, 
                         edge_region,
                         l3_range=None, 
                         l2_range=None,
                         post_edge_range=None,
                         poly_order=2,
                         use_two_window=True,
                         force_flat_post_edge=True,
                         color='#1f77b4'):
    """
    Create a custom element configuration dictionary.
    
    Parameters:
    ----------
    element_name : str
        Name of the element
    edge_range : list
        [min, max] energy range for extraction
    pre_edge_range : list
        [min, max] energy range for pre-edge background fitting
    edge_region : list
        [min, max] energy range containing the actual edge
    l3_range : list, optional
        [min, max] energy range for L3 edge (for plotting)
    l2_range : list, optional
        [min, max] energy range for L2 edge (for plotting)
    post_edge_range : list, optional
        [min, max] energy range for post-edge fitting (if use_two_window=True)
    poly_order : int
        Polynomial order for background fitting
    use_two_window : bool
        Whether to use two-window fitting (pre + post edge)
    force_flat_post_edge : bool
        Whether to force flat baseline after the edge
    color : str
        Color for plotting
        
    Returns:
    -------
    dict
        Element configuration dictionary
    """
    config = {
        'edge_range': edge_range,
        'pre_edge_range': pre_edge_range,
        'edge_region': edge_region,
        'poly_order': poly_order,
        'color': color,
        'use_two_window': use_two_window,
        'force_flat_post_edge': force_flat_post_edge
    }
    
    # Add optional parameters
    if l3_range is not None:
        config['l3_range'] = l3_range
    if l2_range is not None:
        config['l2_range'] = l2_range
    if post_edge_range is not None:
        config['post_edge_range'] = post_edge_range
    elif use_two_window:
        # Auto-generate post-edge range if not provided
        config['post_edge_range'] = [edge_region[1] + 5, edge_range[1]]
    
    return config

def get_available_elements():
    """Return list of elements with default parameters"""
    return list(default_element_params.keys())

def print_element_params(element):
    """Print the parameters for a given element"""
    if element in default_element_params:
        print(f"\nParameters for {element}:")
        for key, value in default_element_params[element].items():
            print(f"  {key}: {value}")
    else:
        print(f"No default parameters available for {element}")

def improved_background_removal(s, element, element_params=None, spectrum_idx=1, show_plots=True):
    """
    Advanced background removal for EELS with flexible element configuration
    
    Parameters:
    ----------
    s : HyperSpy signal
        Original EELS signal
    element : str
        Element name to process
    element_params : dict, optional
        Custom element parameters. If None, uses default parameters for known elements.
        Should contain keys: 'edge_range', 'pre_edge_range', 'edge_region', etc.
    spectrum_idx : int or tuple
        Index of spectrum to process and plot (tuple of (row, column) or int for column with row 0)
    show_plots : bool
        Whether to show diagnostic plots
        
    Returns:
    -------
    element_signal : background-removed element edge signal
    """
    
    # Use provided parameters or fall back to defaults
    if element_params is None:
        if element in default_element_params:
            params = default_element_params[element].copy()
            print(f"Using default parameters for {element}")
        else:
            raise ValueError(f"Element '{element}' not found in defaults. Please provide element_params.")
    else:
        params = element_params.copy()
        print(f"Using custom parameters for {element}")
    
    # Validate required parameters
    required_keys = ['edge_range', 'pre_edge_range', 'edge_region']
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
    
    # Set default values for optional parameters
    params.setdefault('poly_order', 2)
    params.setdefault('use_two_window', True)
    params.setdefault('force_flat_post_edge', True)
    params.setdefault('color', '#1f77b4')
    
    # Extract element edge region
    element_edge = s.isig[params['edge_range'][0]:params['edge_range'][1]].deepcopy()
    
    # Get energy axis
    energy_axis = element_edge.axes_manager.signal_axes[0].axis
    
    # Create output signal
    element_clean = element_edge.deepcopy()
    
    # Get the shape of the data
    data_shape = element_edge.data.shape
    num_rows = data_shape[0]
    num_cols = data_shape[1]
    
    # Check if spectrum_idx is a tuple
    if isinstance(spectrum_idx, tuple) and len(spectrum_idx) == 2:
        row_idx, col_idx = spectrum_idx
    else:
        # For backward compatibility, treat as column index and use row 0
        row_idx, col_idx = 0, spectrum_idx
    
    # Make sure indices are within bounds
    row_idx = min(row_idx, num_rows - 1)
    col_idx = min(col_idx, num_cols - 1)
    
    # Process each spectrum for all rows and columns
    for i in range(num_rows):
        for j in range(num_cols):
            # Get original data for this spectrum
            original = element_edge.data[i, j].copy()
            
            # Define pre-edge region
            pre_edge_mask = (energy_axis >= params['pre_edge_range'][0]) & (energy_axis <= params['pre_edge_range'][1])
            pre_edge_E = energy_axis[pre_edge_mask]
            pre_edge_I = original[pre_edge_mask]
            
            # For two-window fitting, also define post-edge region
            if params['use_two_window'] and 'post_edge_range' in params:
                post_edge_mask = (energy_axis >= params['post_edge_range'][0]) & (energy_axis <= params['post_edge_range'][1])
                post_edge_E = energy_axis[post_edge_mask]
                post_edge_I = original[post_edge_mask]
                
                # Combine pre and post regions for fitting
                fit_E = np.concatenate((pre_edge_E, post_edge_E))
                fit_I = np.concatenate((pre_edge_I, post_edge_I))
            else:
                fit_E = pre_edge_E
                fit_I = pre_edge_I
            
            # Fit a polynomial to the fitting regions
            poly_order = params['poly_order']
            coeffs = np.polyfit(fit_E, fit_I, poly_order)
            
            # Calculate background over full energy range
            background = np.polyval(coeffs, energy_axis)
            
            # Apply smoothing to background
            background = savgol_filter(background, 15, 3)
            
            # Ensure background doesn't exceed signal in edge region to avoid negative values
            edge_region = (energy_axis >= params['edge_region'][0]) & (energy_axis <= params['edge_region'][1])
            for idx in np.where(edge_region)[0]:
                # Cap background at 99% of signal to avoid artifacts
                if background[idx] > original[idx]:
                    background[idx] = original[idx] * 0.99
            
            # Subtract background
            background_removed = original - background
            
            # For elements that need post-edge flattening
            if params['force_flat_post_edge'] and 'l2_range' in params:
                # Find index right after L2 edge
                post_l2_start_idx = np.argmin(np.abs(energy_axis - (params['l2_range'][1] + 5)))
                post_l2_end_idx = len(energy_axis) - 1
                
                if post_l2_start_idx < post_l2_end_idx:
                    # Calculate average value in the post-L2 region
                    post_l2_avg = np.mean(background_removed[post_l2_start_idx:post_l2_end_idx])
                    
                    # Create a gradual transition to zero
                    transition_length = min(20, post_l2_end_idx - post_l2_start_idx)
                    if transition_length > 0:
                        transition = np.linspace(post_l2_avg, 0, transition_length)
                        background_removed[post_l2_start_idx:post_l2_start_idx+transition_length] -= transition
                    
                    # Set all values after transition to zero
                    if post_l2_start_idx + transition_length < post_l2_end_idx:
                        background_removed[post_l2_start_idx+transition_length:] = 0
            
            # Apply minimal smoothing to reduce noise while preserving peak shapes
            background_removed = savgol_filter(background_removed, 7, 3)
            
            # Ensure no negative values
            background_removed = np.maximum(background_removed, 0)
            
            # Store in output signal
            element_clean.data[i, j] = background_removed
            
            # Diagnostic plots for specified spectrum
            if show_plots and i == row_idx and j == col_idx:
                plt.figure(figsize=(7, 5))
                
                # Plot original data and fitted background
                plt.plot(energy_axis, original, 'r-', alpha=0.6, label='Original data')
                plt.plot(energy_axis, background, 'b--', alpha=0.7, label='Fitted background')
                plt.plot(energy_axis, background_removed, 'k-', linewidth=1, label='Background removed')
                
                # Highlight pre-edge region used for fitting
                plt.fill_between(pre_edge_E, 0, np.max(original), color='yellow', alpha=0.1, label='Pre-edge fitting region')
                
                # Highlight post-edge region if used
                if params['use_two_window'] and 'post_edge_range' in params:
                    post_edge_mask = (energy_axis >= params['post_edge_range'][0]) & (energy_axis <= params['post_edge_range'][1])
                    post_edge_E_plot = energy_axis[post_edge_mask]
                    plt.fill_between(post_edge_E_plot, 0, np.max(original), color='orange', alpha=0.1, label='Post-edge fitting region')
                
                # Highlight L3 and L2 regions if provided
                # Use consistent colors for L3 and L2 across all elements
                l3_color = '#FF6B6B'  # Light red for all L3 regions
                l2_color = '#4ECDC4'  # Light teal for all L2 regions
                
                if 'l3_range' in params:
                    l3_mask = (energy_axis >= params['l3_range'][0]) & (energy_axis <= params['l3_range'][1])
                    if np.any(l3_mask):
                        plt.fill_between(energy_axis[l3_mask], 0, np.max(original), 
                                        color=l3_color, alpha=0.15, 
                                        label=f'{element} L3 region')
                
                if 'l2_range' in params:
                    l2_mask = (energy_axis >= params['l2_range'][0]) & (energy_axis <= params['l2_range'][1])
                    if np.any(l2_mask):
                        plt.fill_between(energy_axis[l2_mask], 0, np.max(original), 
                                        color=l2_color, alpha=0.15, 
                                        label=f'{element} L2 region')
                
                # Customize plot
                plt.xlabel('Energy Loss (eV)', fontsize=12)
                plt.ylabel('Intensity', fontsize=12)
                plt.title(f'{element} Edge Background Removal (Row {i}, Col {j})', fontsize=14)
                plt.legend()
                plt.show()
                
                # Plot just the background-removed spectrum
                plt.figure(figsize=(7, 5))
                plt.plot(energy_axis, background_removed, 'k-', linewidth=1)
                plt.xlim(params['edge_region'][0], params['edge_region'][1])  # Focus on edge region
                plt.ylim(bottom=0)  # Force y-axis to start from 0
                plt.xlabel('Energy Loss (eV)', fontsize=12)
                plt.ylabel('Intensity', fontsize=12)
                plt.title(f'Background-Removed {element} Edge Spectrum (Row {i}, Col {j})', fontsize=14)
                plt.show()
    
    return element_clean

def process_multiple_elements(s, elements_config, spectrum_idx=1, show_plots=True, normalize=True):
    """
    Process multiple elements with flexible configuration
    
    Parameters:
    ----------
    s : HyperSpy signal
        Original EELS signal
    elements_config : dict
        Dictionary where keys are element names and values are either:
        - None (use default parameters)
        - dict with custom parameters
        Example: {'Ni': None, 'Co': custom_co_params, 'MyElement': custom_params}
    spectrum_idx : int or tuple
        Index of spectrum to process and plot
    show_plots : bool
        Whether to show diagnostic plots
    normalize : bool
        Whether to normalize spectra to max intensity
        
    Returns:
    -------
    dict containing processed signals for all elements
    """
    results = {}
    
    for element_name, element_params in elements_config.items():
        print(f'Processing: {element_name}')
        
        try:
            processed_signal = improved_background_removal(
                s, 
                element_name, 
                element_params=element_params,
                spectrum_idx=spectrum_idx, 
                show_plots=show_plots
            )
            
            # Normalize if requested
            if normalize:
                processed_signal = processed_signal / np.max(processed_signal.data)
            
            results[element_name] = processed_signal
            
        except Exception as e:
            print(f"Error processing {element_name}: {str(e)}")
            continue
    
    print('Done processing all elements')
    return results

def process_nmc811_elements(s, spectrum_idx=1, show_plots=True, normalize=True):
    """
    Process all elements in NMC811 sample (legacy function for backward compatibility)
    
    Parameters:
    ----------
    s : HyperSpy signal
        Original EELS signal
    spectrum_idx : int or tuple
        Index of spectrum to process and plot. If int, treated as column index for row 0.
        If tuple, should be (row_idx, col_idx).
    show_plots : bool
        Whether to show diagnostic plots
    normalize : bool
        Whether to normalize spectra to max intensity
        
    Returns:
    -------
    dict containing processed signals for all elements
    """
    elements_config = {
        'Ni': None,
        'Co': None,
        'Mn': None,
        'O': None
    }
    
    return process_multiple_elements(s, elements_config, spectrum_idx, show_plots, normalize)