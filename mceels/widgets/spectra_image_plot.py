"""
Interactive widget for visualizing and analyzing EELS spectra images.

This module provides an interactive Jupyter widget for exploring EELS (Electron Energy Loss
Spectroscopy) data. It displays a spatial heatmap alongside individual spectra with Gaussian
fitting capabilities.
"""

from typing import Tuple
from exspy.signals import EELSSpectrum
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, FloatRangeSlider, FloatSlider, VBox, HBox, Output, Label
from IPython.display import display

from mceels.analysis import (
    get_x_array,
    get_y_array,
    get_signal_x_axis,
    fit_double_gaussian_single_spectum,
    gauss,
    double_gauss,
    get_cropped_x_axis,
)

class SpectraImageWidget:
    """
    Interactive widget for EELS spectra visualization and analysis.

    This widget displays two side-by-side plots:
    - Left: A greyscale heatmap showing the sum of intensities across the energy axis
    - Right: The spectrum at a selected position with double Gaussian fitting

    The widget provides interactive sliders to:
    - Navigate through different spatial positions (X and Y indices)
    - Adjust the signal range for Gaussian fitting
    - Control the peak range parameter for fitting

    Attributes:
        _spectra: The EELS spectrum data
        _x_slider: Slider for selecting X position
        _y_slider: Slider for selecting Y position
        _signal_range_slider: Range slider for energy window selection
        _peak_range_slider: Slider for peak range parameter
        _output: Output widget for displaying plots

    Example:
        >>> from mceels.widgets import SpectraImageWidget
        >>> widget = SpectraImageWidget(spectra)
        >>> widget.show()
    """

    def __init__(self, spectra: EELSSpectrum):
        """
        Initialize the SpectraImageWidget.

        Args:
            spectra: An EELSSpectrum object containing the EELS data to visualize.
        """
        self._spectra = spectra

        # Get dimensions
        y_size, x_size = self._spectra.data.shape[0], self._spectra.data.shape[1]

        # Get energy axis range
        energy_axis: np.ndarray = get_signal_x_axis(self._spectra)
        energy_min: float = float(energy_axis[0])
        energy_max: float = float(energy_axis[-1])
        energy_range: float = energy_max - energy_min

        # Create position sliders
        self._x_slider: IntSlider = self._create_slider(max=x_size - 1, description='X index:')
        self._y_slider: IntSlider = self._create_slider(max=y_size - 1, description='Y index:')

        # Create signal range slider (for energy range selection)
        self._signal_range_slider: FloatRangeSlider = FloatRangeSlider(
            value=[850, 870],
            min=energy_min,
            max=energy_max,
            step=0.1,
            description='Signal range:',
            continuous_update=False,
            readout_format='.1f'
        )

        # Create peak range slider
        self._peak_range_slider: FloatSlider = FloatSlider(
            value=20,
            min=0,
            max=energy_range / 2,
            step=0.1,
            description='Peak range:',
            continuous_update=False,
            readout_format='.1f'
        )

        # Create output widget for plots
        self._output: Output = Output()

        # Connect slider events
        self._x_slider.observe(self._on_slider_change, names='value')
        self._y_slider.observe(self._on_slider_change, names='value')
        self._signal_range_slider.observe(self._on_slider_change, names='value')
        self._peak_range_slider.observe(self._on_slider_change, names='value')

    def show(self) -> None:
        """
        Display the interactive widget in a Jupyter notebook.

        This method creates and displays the control sliders for position (X, Y) and
        fitting parameters (signal range, peak range), along with two side-by-side plots
        showing the spatial heatmap and spectrum.

        Note:
            Requires %matplotlib inline or %matplotlib widget in Jupyter notebooks.
        """
        # Create the initial plot
        self._update_plot()

        # Display sliders and plot
        position_controls = VBox([
            Label('Position controls:'),
            self._x_slider,
            self._y_slider
        ])

        fitting_controls = VBox([
            Label('Fitting parameters:'),
            self._signal_range_slider,
            self._peak_range_slider
        ])

        all_controls = HBox([position_controls, fitting_controls])

        display(all_controls)
        display(self._output)

    def _create_slider(self, max: int, description: str) -> IntSlider:
        """
        Create an integer slider widget.

        Args:
            max: Maximum value for the slider.
            description: Label text displayed next to the slider.

        Returns:
            Configured IntSlider widget with range [0, max].
        """
        return IntSlider(
            value=0,
            min=0,
            max=max,
            step=1,
            description=description,
            continuous_update=False
        )

    def _on_slider_change(self, change) -> None:
        """
        Handle slider value changes.

        This callback is triggered when any slider value changes, causing the plot to update.

        Args:
            change: Event object containing information about the change.
        """
        self._update_plot()

    def _update_plot(self) -> None:
        """
        Update the entire plot with current slider values.

        This method clears the output area and redraws both the spatial heatmap
        and the spectrum plot with the current slider values. A red crosshair
        marker is added to the heatmap to indicate the selected position.
        """
        with self._output:
            self._output.clear_output(wait=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot image
            self._plot_image(ax1)

            # Add marker to show selected position
            x_axis: np.ndarray = get_x_array(self._spectra)
            y_axis: np.ndarray = get_y_array(self._spectra)
            x_pos: float = x_axis[self._x_slider.value]
            y_pos: float = y_axis[self._y_slider.value]
            ax1.plot(x_pos, y_pos, 'r+', markersize=15, markeredgewidth=2)

            # Plot spectrum
            self._plot_spectrum(ax2)

            plt.tight_layout()
            plt.show()

    def _plot_image(self, ax: plt.Axes) -> None:
        """
        Plot the spatial heatmap on the given axes.

        Creates a greyscale heatmap showing the sum of intensities across the energy axis
        for each spatial position. The axes are scaled to physical coordinates.

        Args:
            ax: Matplotlib axes object to plot on.
        """
        # Sum intensities across energy axis
        image_to_plot: np.ndarray = np.sum(self._spectra.data, axis=2)

        # Get physical coordinate axes
        x_axis: np.ndarray = get_x_array(self._spectra)
        y_axis: np.ndarray = get_y_array(self._spectra)

        # Define extent for proper axis scaling
        extent: list[float] = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]

        # Display image
        im = ax.imshow(image_to_plot, cmap='gray', aspect='auto', extent=extent, origin='upper')
        ax.set_title('EELS Spectra Image (Sum across energy axis)')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')

    def _plot_spectrum(self, ax: plt.Axes) -> None:
        """
        Plot the spectrum at the selected position with Gaussian fitting.

        This method extracts the spectrum at the current slider position, fits a double
        Gaussian model to the data, and plots the original data along with the fitted
        curve and individual Gaussian components. Vertical lines indicate the cropped
        region used for fitting.

        Args:
            ax: Matplotlib axes object to plot on.

        The plot includes:
            - Blue solid line: Original spectrum data
            - Green dashed line: Double Gaussian fit
            - Magenta dashed line: First Gaussian component
            - Orange dashed line: Second Gaussian component
            - Red dotted lines: Boundaries of the cropped region
        """
        # Get energy axis and spectrum data
        x: np.ndarray = get_signal_x_axis(self._spectra)
        y: np.ndarray = self._spectra.data[self._y_slider.value, self._x_slider.value, :]

        # Get fitting parameters from sliders
        signal_range: Tuple[float, float] = tuple(self._signal_range_slider.value)
        peak_range: float = self._peak_range_slider.value

        # Fit double Gaussian to the spectrum
        double_guass_coeffs = fit_double_gaussian_single_spectum(
            y_axis=y,
            x_axis=x,
            signal_range=signal_range,
            peak_range=peak_range
        )

        # Get the cropped x-axis range used for fitting
        cropped_x: np.ndarray = get_cropped_x_axis(
            y_axis=y,
            x_axis=x,
            signal_range=signal_range,
            peak_range=peak_range
        )

        # Calculate fitted curves
        y_double_guass = double_gauss(x, *double_guass_coeffs)
        y_gauss_1 = gauss(x, *double_guass_coeffs[:3])  # First Gaussian component
        y_gauss_2 = gauss(x, *double_guass_coeffs[3:])  # Second Gaussian component

        # Plot data and fits
        ax.plot(x, y, linestyle='solid', color='blue', label='Data')
        ax.plot(x, y_double_guass, linestyle='dashed', color='green', label='Double Gaussian fit')
        ax.plot(x, y_gauss_1, linestyle='dashed', color='magenta', label='Gaussian 1')
        ax.plot(x, y_gauss_2, linestyle='dashed', color='orange', label='Gaussian 2')

        # Add vertical lines for cropped range
        ax.axvline(x=cropped_x[0], color='red', linestyle='dotted', linewidth=1.5, label='Crop range')
        ax.axvline(x=cropped_x[-1], color='red', linestyle='dotted', linewidth=1.5)

        ax.set_title(f'Spectrum at position ({self._y_slider.value}, {self._x_slider.value})')
        ax.set_xlabel('Energy Loss (eV)')
        ax.set_ylabel('Intensity')
        ax.legend()