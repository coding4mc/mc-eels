"""
Plotting functions for EELS data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_element_centers_flexible(element_data, 
                                 distances=None,
                                 figsize=None,
                                 ylim_ranges=None,
                                 auto_ylim_buffer=1.0,
                                 ncols=2,
                                 title="Element Edge Centers",
                                 show_stats=True,
                                 colors=None,
                                 markersize=4):  # Added markersize parameter
    """
    Plot element centers with flexible number of elements.
    
    Parameters:
    ----------
    element_data : dict
        Dictionary with element names as keys and center data as values
        Example: {'Ni L3': centre_ni_l3, 'Ni L2': centre_ni_l2, 'Co L3': centre_co_l3}
    distances : array-like, optional
        Distance array for x-axis (if None, uses indices)
    figsize : tuple, optional
        Figure size (if None, auto-calculated based on number of elements)
    ylim_ranges : dict, optional
        Dictionary with element names as keys and [min, max] ranges as values
        Example: {'Ni L3': [854, 860], 'Co L3': [777, 786]}
    auto_ylim_buffer : float
        Buffer around data range when auto-scaling (in eV)
    ncols : int
        Number of columns in the subplot grid
    title : str
        Main title for the figure
    show_stats : bool
        Whether to show statistics boxes on plots
    colors : dict, optional
        Custom colors for each element
    markersize : int or float
        Size of the markers (default: 8, was 4)
        
    Returns:
    -------
    fig, axes : matplotlib objects
        
    Example:
    -------
    >>> # Plot just Ni edges with bigger markers
    >>> element_data = {
    ...     'Ni L3': centre_ni_l3,
    ...     'Ni L2': centre_ni_l2
    ... }
    >>> fig, axes = plot_element_centers_flexible(element_data, markersize=10)
    >>> 
    >>> # Plot all elements with custom ranges
    >>> element_data = {
    ...     'Ni L3': centre_ni_l3, 'Ni L2': centre_ni_l2,
    ...     'Co L3': centre_co_l3, 'Co L2': centre_co_l2,
    ...     'Mn L3': centre_mn_l3, 'Mn L2': centre_mn_l2,
    ...     'O Pre-peak': centre_O1, 'O Main': centre_O2
    ... }
    >>> ylim_ranges = {
    ...     'Ni L3': [854, 860], 'Ni L2': [871, 878],
    ...     'Co L3': [777, 786]  # Only specify ranges you want to customize
    ... }
    >>> fig, axes = plot_element_centers_flexible(element_data, 
    ...                                          ylim_ranges=ylim_ranges,
    ...                                          markersize=8)
    """
    
    # Validate input
    if not element_data:
        print("❌ No element data provided")
        return None, None
    
    # Create distance array if not provided
    first_data = list(element_data.values())[0]
    if distances is None:
        distances = np.arange(len(first_data))
        x_label = 'Spectrum Index'
    else:
        x_label = 'Distance (nm)'
    
    # Calculate subplot layout
    n_elements = len(element_data)
    nrows = int(np.ceil(n_elements / ncols))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        width = 5 * ncols
        height = 2 * nrows
        figsize = (width, height)
    
    # Set up default colors
    default_colors = {
        'Ni L3': '#006400', 'Ni L2': '#9BCF78',
        'Co L3': '#00008B', 'Co L2': '#659ADE', 
        'Mn L3': '#ff7f00', 'Mn L2': '#FDB761',
        'O Pre-peak': '#A685BF', 'O Main': '#6a3d9a',
        'O1': '#A685BF', 'O2': '#6a3d9a',  # Alternative names
    }
    
    # Use custom colors if provided, otherwise use defaults or cycle through colors
    if colors is None:
        colors = {}
    
    color_cycle = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # Handle single subplot case
    if n_elements == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each element
    plot_idx = 0
    for element_name, data in element_data.items():
        ax = axes[plot_idx]
        
        # Get color for this element
        if element_name in colors:
            color = colors[element_name]
        elif element_name in default_colors:
            color = default_colors[element_name]
        else:
            color = color_cycle[plot_idx % len(color_cycle)]
        
        # Plot the data with bigger markers
        ax.plot(distances, data, 'o-', color=color, linewidth=2, markersize=markersize)
        ax.set_title(f'{element_name} Edge Center', fontweight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Energy (eV)')
        
        # Set y-axis limits
        if ylim_ranges and element_name in ylim_ranges:
            ax.set_ylim(ylim_ranges[element_name])
            print(f"✅ {element_name}: Custom y-axis range {ylim_ranges[element_name][0]}-{ylim_ranges[element_name][1]} eV")
        else:
            # Auto-scale with buffer
            data_min, data_max = np.min(data), np.max(data)
            y_range = data_max - data_min
            buffer = max(auto_ylim_buffer, y_range * 0.1)
            ax.set_ylim([data_min - buffer, data_max + buffer])
        
        # Add statistics text if requested
        if show_stats:
            stats_text = f'Mean: {np.mean(data):.2f} eV\nStd: {np.std(data):.2f} eV'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Plotted {n_elements} element centers")
    
    return fig, axes

def plot_area_ratio_with_oxygen_dE_updated(distances, 
                                          # Element data (optional - pass None to skip)
                                          area_ratio_mn=None, 
                                          area_ratio_co=None, 
                                          area_ratio_ni=None,
                                          # Oxygen data (optional)
                                          centre_O1=None, 
                                          centre_O2=None,
                                          # Styling
                                          figsize=(10, 6),
                                          plot_title='Compositional Analysis',
                                          # Custom y-axis controls
                                          ratio_ylim=None,        # e.g., [1.5, 4.0]
                                          delta_e_ylim=None,      # e.g., [8.0, 12.0]
                                          markersize=8):
    """
    Plot L3/L2 area ratios for selected elements and optional oxygen ΔE.
    
    Parameters:
    ----------
    distances : array-like
        Distance array for x-axis
    area_ratio_mn, area_ratio_co, area_ratio_ni : array-like or None
        L3/L2 area ratios for each element. Pass None to skip element.
    centre_O1, centre_O2 : array-like or None
        Oxygen pre-peak and main peak centers. Pass None to skip oxygen ΔE.
    figsize : tuple
        Figure size (default: (10, 6))
    plot_title : str
        Main title for the figure
    ratio_ylim : list [min, max], optional
        Y-axis limits for area ratios
    delta_e_ylim : list [min, max], optional
        Y-axis limits for oxygen ΔE
    markersize : int or float
        Size of the markers (default: 8)
        
    Returns:
    -------
    fig, ax1, ax2 : matplotlib objects (ax2 is None if no oxygen data)
        
    Example:
    -------
    >>> # Plot just Ni and Co
    >>> fig, ax1, ax2 = plot_area_ratio_with_oxygen_dE_updated(
    ...     distances, 
    ...     area_ratio_mn=None,        # Skip Mn
    ...     area_ratio_co=area_ratio_co, 
    ...     area_ratio_ni=area_ratio_ni,
    ...     centre_O1=None, centre_O2=None  # Skip oxygen
    ... )
    >>> 
    >>> # Plot all elements with oxygen
    >>> fig, ax1, ax2 = plot_area_ratio_with_oxygen_dE_updated(
    ...     distances, area_ratio_mn, area_ratio_co, area_ratio_ni, 
    ...     centre_O1, centre_O2
    ... )
    """
    
    # Count how many elements we're plotting
    elements = []
    if area_ratio_mn is not None:
        elements.append(('Mn L3/L2', area_ratio_mn, '#ff7f00', 'o'))
    if area_ratio_co is not None:
        elements.append(('Co L3/L2', area_ratio_co, '#00008B', 's'))
    if area_ratio_ni is not None:
        elements.append(('Ni L3/L2', area_ratio_ni, '#006400', '^'))
    
    # Check if we have oxygen data
    plot_oxygen = (centre_O1 is not None and centre_O2 is not None)
    
    if not elements and not plot_oxygen:
        print("❌ No data provided to plot")
        return None, None, None
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = None
    
    # Plot element ratios if we have any
    if elements:
        for label, data, color, marker in elements:
            ax1.plot(distances, data, marker, color=color, label=label, 
                     linewidth=2, markersize=markersize)
        
        ax1.set_ylabel('L3/L2 Area Ratio', fontweight='bold', color='black')
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Set y-axis limits for ratios
        if ratio_ylim is not None:
            ax1.set_ylim(ratio_ylim)
    
    # Plot oxygen ΔE if requested
    if plot_oxygen:
        oxygen_delta_e = centre_O2 - centre_O1
        
        if elements:
            # Create second y-axis if we have both ratios and oxygen
            ax2 = ax1.twinx()
            ax2.plot(distances, oxygen_delta_e, 'x', color='red', label='O ΔE', 
                     markersize=markersize, linestyle='None')
            ax2.set_ylabel('Oxygen ΔE (eV)', fontweight='bold', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            
            # Set y-axis limits for oxygen
            if delta_e_ylim is not None:
                ax2.set_ylim(delta_e_ylim)
        else:
            # Only oxygen data, use main axis
            ax1.plot(distances, oxygen_delta_e, 'x', color='red', label='O ΔE', 
                     markersize=markersize, linestyle='None')
            ax1.set_ylabel('Oxygen ΔE (eV)', fontweight='bold', color='red')
            ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
            
            # Set y-axis limits for oxygen
            if delta_e_ylim is not None:
                ax1.set_ylim(delta_e_ylim)
    
    # Set common labels and title
    ax1.set_xlabel('Distance (nm)', fontweight='bold')
    fig.suptitle(plot_title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print what was plotted
    element_names = [elem[0] for elem in elements]
    if element_names:
        print(f"✅ Plotted elements: {', '.join(element_names)}")
    if plot_oxygen:
        print(f"✅ Plotted oxygen ΔE")
    
    return fig, ax1, ax2