"""
Background subtraction functions for EELS data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def improved_background_removal(s, element, spectrum_idx=1, show_plots=True):
    """
    Advanced background removal for EELS with two-window fitting option.
    
    Parameters:
    ----------
    s : HyperSpy signal
        Original EELS signal
    element : str
        Element to process: 'Mn', 'Co', 'Ni', or 'O'
    spectrum_idx : int
        Index of spectrum to process and plot
    show_plots : bool
        Whether to show diagnostic plots
        
    Returns:
    -------
    element_signal : background-removed element edge signal
    """
    # Define energy ranges and parameters for each element
    element_params = {
        'Ni': {
            'edge_range': [820.0, 900.0],       # Extraction range
            'pre_edge_range': [790.0, 850.0],   # Pre-edge fitting range
            'post_edge_range': [885.0, 900.0],  # Post-edge fitting range (for two-window fitting)
            'edge_region': [850.0, 885.0],      # Actual edge region (L3 + L2)
            'poly_order': 3,                    # Higher polynomial order for Ni
            'l3_range': [853.0, 861.0],         # L3 edge range
            'l2_range': [871.0, 879.0],         # L2 edge range
            'color': '#00008B',                 # Color for plots
            'use_two_window': True,             # Use two-window fitting for Ni
            'force_flat_post_edge': True        # Force flat baseline after L2 edge
        },
        'Co': {
            'edge_range': [720.0, 820.0],
            'pre_edge_range': [720.0, 770.0],
            'post_edge_range': [805, 820.0],
            'edge_region': [775.0, 805.0],
            'poly_order': 2,
            'l3_range': [775, 790.0],
            'l2_range': [792, 800.0],
            'color': '#00008B',
            'use_two_window': True,             # Enable two-window fitting
            'force_flat_post_edge': True        # Enable flat baseline
        },
        'Mn': {
            'edge_range': [590.0, 690.0],
            'pre_edge_range': [580.0, 635.0],
            'post_edge_range': [662.0, 680.0],
            'edge_region': [638.0, 660.0],
            'poly_order': 2,
            'l3_range': [640.0, 646.0],
            'l2_range': [650.0, 659.0],
            'color': '#00008B',
            'use_two_window': True,             # Enable two-window fitting
            'force_flat_post_edge': True        # Enable flat baseline
        },
        'O': {
            'edge_range': [450.0, 590.0],
            'pre_edge_range': [460.0, 510.0],   # Pre-edge fitting range
            'post_edge_range': [556.0, 580.0],  # Post-edge fitting range (3eV after main-peak)
            'edge_region': [525.0, 560.0],      # Edge region (unchanged)
            'poly_order': 4,                    # Polynomial order for fitting
            'l3_range': [525.0, 534.0],         # O pre-peak region
            'l2_range': [535.5, 553.0],         # O main peak region
            'color': '#00008B',                 # Color for plots
            'use_two_window': True,             # Use two-window fitting for O
            'force_flat_post_edge': False       # Don't force flat baseline for O
        }
    }
    
    # Check if element is valid
    if element not in element_params:
        raise ValueError(f"Element '{element}' not supported. Choose from 'Mn', 'Co', 'Ni', or 'O'")
    
    # Get parameters for the selected element
    params = element_params[element]
    
    # Extract element edge region
    element_edge = s.isig[params['edge_range'][0]:params['edge_range'][1]].deepcopy()
    
    # Get energy axis
    energy_axis = element_edge.axes_manager.signal_axes[0].axis
    
    # Create output signal
    element_clean = element_edge.deepcopy()
    
    # Process each spectrum
    for row in range(element_edge.data.shape[0]):
        for col in range(element_edge.data.shape[1]):
            # Get original data for this spectrum
            original = element_edge.data[row, col].copy()
            
            # Define pre-edge region
            pre_edge_mask = (energy_axis >= params['pre_edge_range'][0]) & (energy_axis <= params['pre_edge_range'][1])
            pre_edge_E = energy_axis[pre_edge_mask]
            pre_edge_I = original[pre_edge_mask]
            
            # For two-window fitting, also define post-edge region
            if params['use_two_window']:
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
            
            # For Ni and other elements that need post-edge flattening
            if params['force_flat_post_edge']:
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
            element_clean.data[row, col] = background_removed
    
    # Diagnostic plots for specified spectrum (only for the first row)
    if show_plots and row == 0 and col == spectrum_idx:
        plt.figure(figsize=(7, 5))
        
        # Plot original data and fitted background
        plt.plot(energy_axis, original, 'r-', alpha=0.6, label='Original data')
        plt.plot(energy_axis, background, 'b--', alpha=0.7, label='Fitted background')
        plt.plot(energy_axis, background_removed, 'k-', linewidth=1, label='Background removed')
        
        # Highlight pre-edge region used for fitting
        plt.fill_between(pre_edge_E, 0, np.max(original), color='yellow', alpha=0.1, label='Pre-edge fitting region')
        
        # Highlight post-edge region if used
        if params['use_two_window']:
            plt.fill_between(post_edge_E, 0, np.max(original), color='orange', alpha=0.1, label='Post-edge fitting region')
        
        # Highlight L3 and L2 regions
        l3_mask = (energy_axis >= params['l3_range'][0]) & (energy_axis <= params['l3_range'][1])
        l2_mask = (energy_axis >= params['l2_range'][0]) & (energy_axis <= params['l2_range'][1])
        
        if np.any(l3_mask):
            plt.fill_between(energy_axis[l3_mask], 0, np.max(original), 
                            color=params['color'], alpha=0.1, 
                            label=f'{element} L3 region' if element != 'O' else 'O pre-peak region')
        
        if np.any(l2_mask):
            plt.fill_between(energy_axis[l2_mask], 0, np.max(original), 
                            color='lightblue', alpha=0.1, 
                            label=f'{element} L2 region' if element != 'O' else 'O main peak region')
        
        # Customize plot
        plt.xlabel('Energy Loss (eV)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.title(f'{element} Edge Background Removal', fontsize=14)
        plt.legend()
        plt.show()
        
        # Plot just the background-removed spectrum
        plt.figure(figsize=(7, 5))
        plt.plot(energy_axis, background_removed, 'k-', linewidth=1)
        plt.xlim(params['edge_region'][0], params['edge_region'][1])  # Focus on edge region
        plt.ylim(bottom=0)  # Force y-axis to start from 0
        plt.xlabel('Energy Loss (eV)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.title(f'Background-Removed {element} Edge Spectrum', fontsize=14)
        plt.show()
    
    return element_clean


def process_nmc811_elements(s, spectrum_idx=1, show_plots=True, normalize=True):
    """
    Process all elements in NMC811 sample.
    
    Parameters:
    ----------
    s : HyperSpy signal
        Original EELS signal
    spectrum_idx : int
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
    
    # Process all elements
    print('Processing: Ni')
    ni = improved_background_removal(s, 'Ni', spectrum_idx, show_plots)
    
    print('Processing: Co')
    co = improved_background_removal(s, 'Co', spectrum_idx, show_plots)
    
    print('Processing: Mn')
    mn = improved_background_removal(s, 'Mn', spectrum_idx, show_plots)
    
    print('Processing: O')
    o = improved_background_removal(s, 'O', spectrum_idx, show_plots)
    
    # Normalize if requested
    if normalize:
        print('Normalizing')
        ni = ni/np.max(ni.data)
        co = co/np.max(co.data)
        mn = mn/np.max(mn.data)
        o = o/np.max(o.data)
    
    print('Done')
    
    # Store results
    results['Ni'] = ni
    results['Co'] = co
    results['Mn'] = mn
    results['O'] = o
    
    return results
