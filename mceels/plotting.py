"""
Plotting functions for EELS data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def color_to_number(color):
    """
    Convert color name to hex color code.
    
    Parameters:
    ----------
    color : str
        Color name
        
    Returns:
    -------
    str
        Hex color code
    """
    color_mapping = {
        'o1': '#FF0000',  # red
        'o2': '#DB7093',  # pink
        'ni_l3': '#006400',  # dark green
        'ni_l2': '#9BCF78',  # light green
        'co_l3': '#00008B',  # dark blue
        'co_l2': '#659ADE',  # light blue
        'mn_l3': '#ff7f00',  # dark orange
        'mn_l2': '#FDB761',  # light orange
        'blue': '#1f78b4',
        'orange': '#ff7f00'
    }
    return color_mapping.get(color.lower(), None)


def plot_area_ratio_with_oxygen_dE_updated(
    distances, area_ratio_mn, area_ratio_co, area_ratio_ni, 
    centre_O1, centre_O2, figsize=(10, 8), plot_title='NMC811 Compositional Analysis'
):
    """
    Create a plot with L3/L2 area ratios and oxygen dE with legend at the bottom.
    
    Parameters:
    ----------
    distances : array-like
        Array of distances or point indices
    area_ratio_mn : array-like
        Mn L3/L2 area ratios
    area_ratio_co : array-like
        Co L3/L2 area ratios
    area_ratio_ni : array-like
        Ni L3/L2 area ratios
    centre_O1 : array-like
        O1 peak centers
    centre_O2 : array-like
        O2 peak centers
    figsize : tuple
        Figure size (width, height)
    plot_title : str
        Custom title for the plot
        
    Returns:
    -------
    fig, ax1, ax2 : Figure and Axes objects
    """
    # Colors for plotting
    ni_color = '#006400'  # dark green
    co_color = '#00008B'  # dark blue
    mn_color = '#ff7f00'  # dark orange
    o_color = '#FF0000'   # red
    
    # Create figure with specifically increased height for legend
    fig = plt.figure(figsize=figsize)
    
    # Add axes, leaving space at the bottom for legend
    # The dimensions are [left, bottom, width, height]
    ax1 = fig.add_axes([0.1, 0.2, 0.8, 0.7])
    
    # Plot L3/L2 area ratios with different markers
    ni_line = ax1.plot(distances, area_ratio_ni, 'o', color=ni_color, 
             label='L3/L2 Ni', markersize=8, linestyle='')
    co_line = ax1.plot(distances, area_ratio_co, 's', color=co_color, 
             label='L3/L2 Co', markersize=8, linestyle='')
    mn_line = ax1.plot(distances, area_ratio_mn, '^', color=mn_color, 
             label='L3/L2 Mn', markersize=8, linestyle='')
    
    # Create second y-axis for oxygen dE
    ax2 = ax1.twinx()
    
    # Calculate oxygen dE
    dE = centre_O2 - centre_O1
    
    # Plot oxygen dE - using just 'x' markers, no line
    o_line = ax2.plot(distances, dE, 'x', color=o_color, 
             label='ΔE O', markersize=8, linestyle='')
    
    # Set labels and title
    ax1.set_xlabel('Distance from surface (nm)', fontsize=12)
    ax1.set_ylabel('L3/L2 Area Ratio', fontsize=12)
    ax2.set_ylabel('ΔE (eV)', fontsize=12, color=o_color)
    ax1.set_title(plot_title, fontsize=14)
    
    # Set color for y-axis on the right to match ΔE color
    ax2.tick_params(axis='y', colors=o_color)
    
    # Remove grid lines
    ax1.grid(False)
    ax2.grid(False)
    
    # Custom legend handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ni_color, markersize=10, label='L3/L2 Ni'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=co_color, markersize=10, label='L3/L2 Co'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=mn_color, markersize=10, label='L3/L2 Mn'),
        Line2D([0], [0], marker='x', color=o_color, markersize=10, label='ΔE O', linestyle='None')
    ]
    
    # Add legend below the plot
    fig.legend(handles=legend_elements,
              loc='lower center', 
              bbox_to_anchor=(0.5, 0.02),
              ncol=4, fontsize=12)
    
    return fig, ax1, ax2


def plot_element_centers_grid(
    centre_ni_l3, centre_ni_l2, 
    centre_co_l3, centre_co_l2, 
    centre_mn_l3, centre_mn_l2,
    centre_O1, centre_O2,
    distances=None
):
    """
    Create a grid of plots showing element center positions (2x4 layout).
    
    Parameters:
    ----------
    centre_ni_l3, centre_ni_l2 : array-like
        Ni L3 and L2 center positions
    centre_co_l3, centre_co_l2 : array-like
        Co L3 and L2 center positions
    centre_mn_l3, centre_mn_l2 : array-like
        Mn L3 and L2 center positions
    centre_O1, centre_O2 : array-like
        O pre-peak and main-peak center positions
    distances : array-like, optional
        Distance array for x-axis. If None, array indices will be used.
        
    Returns:
    -------
    fig, axes : Figure and Axes objects
    """
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True)
    
    # Colors for plotting - define them explicitly
    ni_l3_color = '#006400'  # dark green
    ni_l2_color = '#9BCF78'  # light green
    co_l3_color = '#00008B'  # dark blue
    co_l2_color = '#659ADE'  # light blue
    mn_l3_color = '#ff7f00'  # dark orange
    mn_l2_color = '#FDB761'  # light orange
    o1_color = '#FF0000'     # red (for O prepeak)
    o2_color = '#DB7093'     # pale violet red (for O main peak)
    
    # Create distance array if not provided
    if distances is None:
        distances = np.arange(len(centre_ni_l3))
    
    # Ni plots
    axes[0, 0].plot(distances, centre_ni_l3, 'o-', color=ni_l3_color)
    axes[0, 0].set_title('Ni L3 Peak Center')
    axes[0, 0].set_ylim(centre_ni_l3.min() - 0.2, centre_ni_l3.max() + 0.2)
    
    axes[0, 1].plot(distances, centre_ni_l2, 'o-', color=ni_l2_color)
    axes[0, 1].set_title('Ni L2 Peak Center')
    axes[0, 1].set_ylim(centre_ni_l2.min() - 0.2, centre_ni_l2.max() + 0.2)
    
    # Co plots
    axes[1, 0].plot(distances, centre_co_l3, 'o-', color=co_l3_color)
    axes[1, 0].set_title('Co L3 Peak Center')
    axes[1, 0].set_ylim(centre_co_l3.min() - 0.2, centre_co_l3.max() + 0.2)
    
    axes[1, 1].plot(distances, centre_co_l2, 'o-', color=co_l2_color)
    axes[1, 1].set_title('Co L2 Peak Center')
    axes[1, 1].set_ylim(centre_co_l2.min() - 0.2, centre_co_l2.max() + 0.2)
    
    # Mn plots
    axes[2, 0].plot(distances, centre_mn_l3, 'o-', color=mn_l3_color)
    axes[2, 0].set_title('Mn L3 Peak Center')
    axes[2, 0].set_ylim(centre_mn_l3.min() - 0.2, centre_mn_l3.max() + 0.2)
    
    axes[2, 1].plot(distances, centre_mn_l2, 'o-', color=mn_l2_color)
    axes[2, 1].set_title('Mn L2 Peak Center')
    axes[2, 1].set_ylim(centre_mn_l2.min() - 0.2, centre_mn_l2.max() + 0.2)
    
    # O plots with updated colors
    axes[3, 0].plot(distances, centre_O1, 'o-', color=o1_color)
    axes[3, 0].set_title('O Pre-peak Center')
    axes[3, 0].set_ylim(centre_O1.min() - 0.2, centre_O1.max() + 0.2)
    
    axes[3, 1].plot(distances, centre_O2, 'o-', color=o2_color)
    axes[3, 1].set_title('O Main-peak Center')
    axes[3, 1].set_ylim(centre_O2.min() - 0.2, centre_O2.max() + 0.2)
    
    # Set common properties for all subplots
    for i in range(4):
        for j in range(2):
            # Remove gridlines
            axes[i, j].grid(False)
            # Set labels
            axes[i, j].set_xlabel('Distance from surface (nm)')
            axes[i, j].set_ylabel('Energy (eV)')
    
    plt.suptitle('Element Edge Positions', fontsize=14)
    
    return fig, axes


def save_all_elements_on_one_figure(
    ni_signal, co_signal, mn_signal, o_signal,
    g12_coeffs_ni_l3, g12_coeffs_ni_l2,
    g12_coeffs_mn_l3, g12_coeffs_mn_l2,
    g12_coeffs_co_l3, g12_coeffs_co_l2,
    g12_coeffs_o_pre, g123_coeffs_o_main,
    spectrum_idx=0,
    output_dir="./Combined_Plots",
    dpi=300,
    show_plot=False,
    batch_process=False
):
    """
    Create a single figure with all 8 plots arranged in 4 rows with reordered elements:
    Row 1: O pre-peak and main-peak
    Row 2: Mn L3 and L2
    Row 3: Co L3 and L2
    Row 4: Ni L3 and L2
    
    Parameters:
    ----------
    ni_signal, co_signal, mn_signal, o_signal : HyperSpy signals
        Signal data for each element
    g12_coeffs_*_* : numpy.ndarray
        Coefficients for double-Gaussian fits for each edge
    g123_coeffs_o_main : numpy.ndarray
        Coefficients for triple-Gaussian fit for O main-peak
    spectrum_idx : int
        Index of the spectrum to plot. If None and batch_process=True, process all spectra.
    output_dir : str
        Directory to save the output figures
    dpi : int
        Resolution for saved figures
    show_plot : bool
        Whether to display the figure
    batch_process : bool
        If True, process all spectra in the dataset
        
    Returns:
    -------
    str or list
        Path to saved figure or list of paths if batch_process=True
    """
    from ..utils.gaussian import gauss, double_gauss, triple_gauss, get_signal_x_axis
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle batch processing
    if batch_process or spectrum_idx is None:
        # Get total number of spectra
        total_spectra = len(g12_coeffs_ni_l3)
        saved_paths = []
        
        print(f"Processing all {total_spectra} spectra in batch mode...")
        
        for idx in range(total_spectra):
            try:
                path = save_all_elements_on_one_figure(
                    ni_signal, co_signal, mn_signal, o_signal,
                    g12_coeffs_ni_l3, g12_coeffs_ni_l2,
                    g12_coeffs_mn_l3, g12_coeffs_mn_l2,
                    g12_coeffs_co_l3, g12_coeffs_co_l2,
                    g12_coeffs_o_pre, g123_coeffs_o_main,
                    spectrum_idx=idx,
                    output_dir=output_dir,
                    dpi=dpi,
                    show_plot=show_plot,
                    batch_process=False  # Prevent recursion
                )
                saved_paths.append(path)
            except Exception as e:
                print(f"Error processing spectrum {idx}: {str(e)}")
        
        print(f"Batch processing complete. Saved {len(saved_paths)} plots to {output_dir}")
        return saved_paths
    
    # If spectrum_idx is still None at this point (error condition), use 0
    if spectrum_idx is None:
        spectrum_idx = 0
        print("Warning: spectrum_idx was None, using spectrum 0 instead.")
    
    # Define custom colors for each element and edge type
    element_colors = {
        'Ni': {'L3': '#006400', 'L2': '#9BCF78'},    # dark green, light green
        'Co': {'L3': '#00008B', 'L2': '#659ADE'},    # dark blue, light blue
        'Mn': {'L3': '#ff7f00', 'L2': '#FDB761'},    # dark orange, light orange
        'O': {'pre-peak': '#FF0000', 'main-peak': '#DB7093'}  # red, pink
    }
    
    # Set line styles for Gaussian components
    g1_color = 'black'         # Black solid line for Gaussian 1
    g1_style = ':'
    g2_color = 'black'         # Black dotted line for Gaussian 2
    g2_style = '-'
    g3_color = 'black'       # Dark Turquoise for the third Gaussian (O main-peak)
    g3_style = '-.'
    
    # Get x-axes for each element
    x_ni = get_signal_x_axis(ni_signal)
    x_mn = get_signal_x_axis(mn_signal)
    x_co = get_signal_x_axis(co_signal)
    x_o = get_signal_x_axis(o_signal)
    
    # Get spectrum data
    ni_data = ni_signal.data[0][spectrum_idx]
    mn_data = mn_signal.data[0][spectrum_idx]
    co_data = co_signal.data[0][spectrum_idx]
    o_data = o_signal.data[0][spectrum_idx]
    
    # Get Gaussian coefficients for the selected spectrum
    ni_l3_coeff = g12_coeffs_ni_l3[spectrum_idx]
    ni_l2_coeff = g12_coeffs_ni_l2[spectrum_idx]
    mn_l3_coeff = g12_coeffs_mn_l3[spectrum_idx]
    mn_l2_coeff = g12_coeffs_mn_l2[spectrum_idx]
    co_l3_coeff = g12_coeffs_co_l3[spectrum_idx]
    co_l2_coeff = g12_coeffs_co_l2[spectrum_idx]
    o_pre_coeff = g12_coeffs_o_pre[spectrum_idx]
    o_main_coeff = g123_coeffs_o_main[spectrum_idx]
    
    # Create figure with 4 rows, 2 columns
    fig = plt.figure(figsize=(10, 12))
    
    # Create grid for subplots with some spacing
    gs = plt.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.2)
    
    # Create axes for each subplot - REORDERED with O at top, then Mn, Co, Ni
    ax_o_pre = fig.add_subplot(gs[0, 0])
    ax_o_main = fig.add_subplot(gs[0, 1])
    ax_mn_l3 = fig.add_subplot(gs[1, 0])
    ax_mn_l2 = fig.add_subplot(gs[1, 1])
    ax_co_l3 = fig.add_subplot(gs[2, 0])
    ax_co_l2 = fig.add_subplot(gs[2, 1]) 
    ax_ni_l3 = fig.add_subplot(gs[3, 0])
    ax_ni_l2 = fig.add_subplot(gs[3, 1])
    
    # Plot Ni L3
    ni_l3_g1 = gauss(x_ni, *ni_l3_coeff[:3])
    ni_l3_g2 = gauss(x_ni, *ni_l3_coeff[3:])
    ni_l3_combined = double_gauss(x_ni, *ni_l3_coeff)
    
    ax_ni_l3.plot(x_ni, ni_data, 'o', color='gray', alpha=0.6, markersize=3)
    ax_ni_l3.plot(x_ni, ni_l3_g1, g1_style, color=g1_color, linewidth=2)
    ax_ni_l3.plot(x_ni, ni_l3_g2, g2_style, color=g2_color, linewidth=2)
    ax_ni_l3.plot(x_ni, ni_l3_combined, '-', color=element_colors['Ni']['L3'], linewidth=2)
    
    # Plot Ni L2
    ni_l2_g1 = gauss(x_ni, *ni_l2_coeff[:3])
    ni_l2_g2 = gauss(x_ni, *ni_l2_coeff[3:])
    ni_l2_combined = double_gauss(x_ni, *ni_l2_coeff)
    
    ax_ni_l2.plot(x_ni, ni_data, 'o', color='gray', alpha=0.6, markersize=3)
    ax_ni_l2.plot(x_ni, ni_l2_g1, g1_style, color=g1_color, linewidth=2)
    ax_ni_l2.plot(x_ni, ni_l2_g2, g2_style, color=g2_color, linewidth=2)
    ax_ni_l2.plot(x_ni, ni_l2_combined, '-', color=element_colors['Ni']['L2'], linewidth=2)
    
    # Plot Mn L3
    mn_l3_g1 = gauss(x_mn, *mn_l3_coeff[:3])
    mn_l3_g2 = gauss(x_mn, *mn_l3_coeff[3:])
    mn_l3_combined = double_gauss(x_mn, *mn_l3_coeff)
    
    ax_mn_l3.plot(x_mn, mn_data, 'o', color='gray', alpha=0.6, markersize=3)
    ax_mn_l3.plot(x_mn, mn_l3_g1, g1_style, color=g1_color, linewidth=2)
    ax_mn_l3.plot(x_mn, mn_l3_g2, g2_style, color=g2_color, linewidth=2)
    ax_mn_l3.plot(x_mn, mn_l3_combined, '-', color=element_colors['Mn']['L3'], linewidth=2)
    
    # Plot Mn L2
    mn_l2_g1 = gauss(x_mn, *mn_l2_coeff[:3])
    mn_l2_g2 = gauss(x_mn, *mn_l2_coeff[3:])
    mn_l2_combined = double_gauss(x_mn, *mn_l2_coeff)
    
    ax_mn_l2.plot(x_mn, mn_data, 'o', color='gray', alpha=0.6, markersize=3)
    ax_mn_l2.plot(x_mn, mn_l2_g1, g1_style, color=g1_color, linewidth=2)
    ax_mn_l2.plot(x_mn, mn_l2_g2, g2_style, color=g2_color, linewidth=2)
    ax_mn_l2.plot(x_mn, mn_l2_combined, '-', color=element_colors['Mn']['L2'], linewidth=2)
    
    # Plot Co L3
    co_l3_g1 = gauss(x_co, *co_l3_coeff[:3])
    co_l3_g2 = gauss(x_co, *co_l3_coeff[3:])
    co_l3_combined = double_gauss(x_co, *co_l3_coeff)
    
    ax_co_l3.plot(x_co, co_data, 'o', color='gray', alpha=0.6, markersize=3)
    ax_co_l3.plot(x_co, co_l3_g1, g1_style, color=g1_color, linewidth=2)
    ax_co_l3.plot(x_co, co_l3_g2, g2_style, color=g2_color, linewidth=2)
    ax_co_l3.plot(x_co, co_l3_combined, '-', color=element_colors['Co']['L3'], linewidth=2)
    
    # Plot Co L2
    co_l2_g1 = gauss(x_co, *co_l2_coeff[:3])
    co_l2_g2 = gauss(x_co, *co_l2_coeff[3:])
    co_l2_combined = double_gauss(x_co, *co_l2_coeff)
    
    ax_co_l2.plot(x_co, co_data, 'o', color='gray', alpha=0.6, markersize=3)
    ax_co_l2.plot(x_co, co_l2_g1, g1_style, color=g1_color, linewidth=2)
    ax_co_l2.plot(x_co, co_l2_g2, g2_style, color=g2_color, linewidth=2)
    ax_co_l2.plot(x_co, co_l2_combined, '-', color=element_colors['Co']['L2'], linewidth=2)
    
    # Plot O pre-peak
    o_pre_g1 = gauss(x_o, *o_pre_coeff[:3])
    o_pre_g2 = gauss(x_o, *o_pre_coeff[3:])
    o_pre_combined = double_gauss(x_o, *o_pre_coeff)
    
    ax_o_pre.plot(x_o, o_data, 'o', color='gray', alpha=0.6, markersize=3)
    ax_o_pre.plot(x_o, o_pre_g1, g1_style, color=g1_color, linewidth=2)
    ax_o_pre.plot(x_o, o_pre_g2, g2_style, color=g2_color, linewidth=2)
    ax_o_pre.plot(x_o, o_pre_combined, '-', color=element_colors['O']['pre-peak'], linewidth=2)
    
    # Plot O main-peak (triple Gaussian)
    o_main_g1 = gauss(x_o, *o_main_coeff[:3])
    o_main_g2 = gauss(x_o, *o_main_coeff[3:6])
    o_main_g3 = gauss(x_o, *o_main_coeff[6:])
    o_main_combined = triple_gauss(x_o, *o_main_coeff)
    
    ax_o_main.plot(x_o, o_data, 'o', color='gray', alpha=0.6, markersize=3)
    ax_o_main.plot(x_o, o_main_g1, g1_style, color=g1_color, linewidth=2)
    ax_o_main.plot(x_o, o_main_g2, g2_style, color=g2_color, linewidth=2)
    ax_o_main.plot(x_o, o_main_g3, g3_style, color=g3_color, linewidth=2)
    ax_o_main.plot(x_o, o_main_combined, '-', color=element_colors['O']['main-peak'], linewidth=2)
    
    # Set axis limits for better focusing on the edges with updated ranges
    # Ni edges
    ax_ni_l3.set_xlim(850, 864)
    ax_ni_l2.set_xlim(870, 880)  # Updated range
    
    # Mn edges
    ax_mn_l3.set_xlim(638, 652)  # Updated range
    ax_mn_l2.set_xlim(646, 662)  # Updated range
    
    # Co edges
    ax_co_l3.set_xlim(776, 792)  # Updated range
    ax_co_l2.set_xlim(792, 806)  # Updated range
    
    # O edges - adjusting based on data content
    ax_o_pre.set_xlim(525, 555)
    ax_o_main.set_xlim(525, 555)
    
    # Add subplot titles
    ax_o_pre.set_title("O pre-peak", fontsize=12)
    ax_o_main.set_title("O main-peak", fontsize=12)
    ax_ni_l3.set_title("Ni L3", fontsize=12)
    ax_ni_l2.set_title("Ni L2", fontsize=12)
    ax_mn_l3.set_title("Mn L3", fontsize=12)
    ax_mn_l2.set_title("Mn L2", fontsize=12)
    ax_co_l3.set_title("Co L3", fontsize=12)
    ax_co_l2.set_title("Co L2", fontsize=12)

    # Add axis labels (only on left and bottom edges)
    for ax in [ax_o_pre, ax_mn_l3, ax_co_l3, ax_ni_l3]:
        ax.set_ylabel("Intensity", fontsize=11)
    
    for ax in [ax_ni_l3, ax_ni_l2]:
        ax.set_xlabel("Energy Loss (eV)", fontsize=11)
    
    # Remove grid from all subplots
    for ax in [ax_ni_l3, ax_ni_l2, ax_mn_l3, ax_mn_l2, ax_co_l3, ax_co_l2, ax_o_pre, ax_o_main]:
        ax.grid(False)

    # Create legend items - MODIFIED to move O peaks to second row
    handles1, labels1 = [], []
    handles2, labels2 = [], []
    
    # First row: Raw data and Gaussian components
    handles1.append(plt.Line2D([0], [0], marker='o', color='gray', alpha=0.6, markersize=4, linestyle='None'))
    labels1.append("Raw data")
    handles1.append(plt.Line2D([0], [0], color=g1_color, linestyle=g1_style, linewidth=2))
    labels1.append("Gaussian 1")
    handles1.append(plt.Line2D([0], [0], color=g2_color, linestyle=g2_style, linewidth=2))
    labels1.append("Gaussian 2")
    handles1.append(plt.Line2D([0], [0], color=g3_color, linestyle=g3_style, linewidth=2))
    labels1.append("Gaussian 3")
    
    # Second row: Metal edges and O peaks
    handles2.append(plt.Line2D([0], [0], color=element_colors['O']['pre-peak'], linewidth=2))
    labels2.append("O pre-peak")
    handles2.append(plt.Line2D([0], [0], color=element_colors['O']['main-peak'], linewidth=2))
    labels2.append("O main-peak")
    handles2.append(plt.Line2D([0], [0], color=element_colors['Mn']['L3'], linewidth=2))
    labels2.append("Mn L3")
    handles2.append(plt.Line2D([0], [0], color=element_colors['Mn']['L2'], linewidth=2))
    labels2.append("Mn L2")
    handles2.append(plt.Line2D([0], [0], color=element_colors['Co']['L3'], linewidth=2))
    labels2.append("Co L3")
    handles2.append(plt.Line2D([0], [0], color=element_colors['Co']['L2'], linewidth=2))
    labels2.append("Co L2")
    handles2.append(plt.Line2D([0], [0], color=element_colors['Ni']['L3'], linewidth=2))
    labels2.append("Ni L3")
    handles2.append(plt.Line2D([0], [0], color=element_colors['Ni']['L2'], linewidth=2))
    labels2.append("Ni L2")

    # Add first row legend
    first_legend = fig.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, 0.035), 
                              ncol=4, fontsize=9, frameon=True)
    plt.gca().add_artist(first_legend)
    
    # Add second row legend
    fig.legend(handles2, labels2, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
               ncol=8, fontsize=9, frameon=True)
    
    # Adjust layout to make room for the legends
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    # Add overall title
    fig.suptitle(f"NMC811 EELS Analysis - Spectrum {spectrum_idx+1}", fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    # Save figure
    filepath = os.path.join(output_dir, f"all_elements_spectrum_{spectrum_idx+1}.png")
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    # Show or close figure
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    print(f"Saved combined plot for spectrum {spectrum_idx+1} to {filepath}")
    return filepath


def plot_axes(axes, x, y, color='#A685BF', legend_name=None, style ='o'):
    def color_to_number(color):
        color_mapping = {
            'o1': '#A685BF',  # light purple
            'o2': '#6a3d9a',  # dark purple
            'ni_l3': '#006400',  # dark green
            'ni_l2': '#9BCF78',  # light green
            'co_l3': '#00008B',  # dark blue
            'co_l2': '#659ADE',  # light blue
            'mn_l3': '#ff7f00',  # dark orange
            'mn_l2': '#FDB761',  # light orange
            'blue': '#1f78b4',
            'orange': '#ff7f00',
            'red': '#a10000'
        }
        return color_mapping.get(color.lower(), None)

    numerical_value = color_to_number(color)

    axes.plot(x, y, style, color=numerical_value, label=legend_name)  # Use numerical_value here


def plot_ratio_vs_distance_new(
    axes, intensity_1, intensity_2, reverse='yes/no', color='#A685BF',
    legend_name=None
):
    # Correct indentation for the if-else block
    if reverse == 'yes':
        intensity_1 = intensity_1[::-1]
        intensity_2 = intensity_2[::-1]
    else:
        intensity_1 = intensity_1
        intensity_2 = intensity_2

    ratio = intensity_1 / intensity_2
    x_axis = np.arange(1, len(ratio) + 1, 1)

    plot_axes(axes, x_axis, ratio, color, legend_name, 'o')


def plot_e(
    axes, center_1, center_2, reverse='yes/no', color='#A685BF',
    legend_name=None
):
    # Correct indentation for the if-else block
    if reverse == 'yes':
        center_1 = center_1[::-1]
        center_2 = center_2[::-1]
    else:
        center_1 = center_1
        center_2 = center_2

    delta_e = center_2 - center_1
    ratio = center_1/center_2
    x_axis = np.arange(1, len(ratio) + 1, 1)

    plot_axes(axes, x_axis, delta_e, color, legend_name, 'x')
