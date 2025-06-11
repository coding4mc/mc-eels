"""
Preprocessing functions for EELS data analysis.
"""

from typing import Tuple
from scipy.ndimage import gaussian_filter
from exspy.signals import EELSSpectrum
import numpy as np


def preprocess(
        signal: EELSSpectrum,
        signal_ll: EELSSpectrum,
        vacuum_x: int,
        blur_data: bool = False,
        keep_left: bool = True
) -> Tuple[EELSSpectrum, EELSSpectrum]:
    """
    Preprocesses the signal by masking the vacuum part of the signal.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to preprocess
    signal_ll : HyperSpy signal
        Low-loss signal
    vacuum_x : float
        X value where vacuum starts
    blur_data : bool
        Whether to apply Gaussian blur
    keep_left : bool
        Whether to keep the left part of the signal (True) or the right part (False)
        
    Returns:
    -------
    tuple
        Preprocessed signal and low-loss signal
    """
    if blur_data:
        signal = gauss_blur_data(signal)
        
    signal = shift(signal)
    signal = normalise(signal)

    # Remove vacuum
    vacuum_index = get_closest_x_index(signal, vacuum_x)

    if keep_left:
        return signal.inav[0:vacuum_index], signal_ll.inav[0:vacuum_index]
    else:
        return signal.inav[vacuum_index:], signal_ll.inav[vacuum_index:]


def rotate(signal: EELSSpectrum, degree: float):
    """
    Rotates the given signal by the given amount of degrees (anti-clockwise).
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to rotate
    degree : float
        Rotation angle in degrees
        
    Returns:
    -------
    HyperSpy signal
        Rotated signal
    """
    s = signal.copy()
    data = s.data
    
    theta = degree * np.pi / 180
 
    rotation_matrix = np.zeros((2, 2))
    rotation_matrix[0] = np.array([np.cos(theta), -np.sin(theta)])
    rotation_matrix[1] = np.array([np.sin(theta), np.cos(theta)])
    
    y, x, z = data.shape
    centre_y, centre_x = y // 2, x // 2
    
    new_data = np.zeros((y, x, z))
    for i0 in range(x):
        for j0 in range(y):
            initial_vector = np.array([i0 - centre_x, j0 - centre_y])
            new_vector = np.matmul(initial_vector, rotation_matrix)
            
            i1, j1 = new_vector
            
            i1 = int(np.round(i1 + centre_x))
            j1 = int(np.round(j1 + centre_y))
            
            if 0 <= i1 < x and 0 <= j1 < y:
                new_data[j1, i1, :] = data[j0, i0]
              
    s.data = new_data
    return fill_empty_spaces(s)


# Helper functions needed by preprocess and rotate
def normalise(signal: EELSSpectrum):
    """
    Normalises the data such that the intensities lie between 0 and 1.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to normalize
        
    Returns:
    -------
    HyperSpy signal
        Normalized signal
    """
    return signal / np.max(signal.data)


def shift(signal):
    """
    Shifts the signal data up such that no data is negative.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to shift
        
    Returns:
    -------
    HyperSpy signal
        Shifted signal
    """
    i_min = np.min(signal.data)
    signal_shifted = signal - i_min
    return signal_shifted


def get_x_array(signal):
    """
    Returns an array of the X-axis of the image.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to get X-axis from
        
    Returns:
    -------
    numpy.ndarray
        X-axis array
    """
    x_scale = signal.axes_manager.navigation_axes[0].scale
    x_size = signal.axes_manager.navigation_axes[0].size
    return np.arange(x_size) * x_scale


def get_closest_x_index(signal, value):
    """
    Returns the index closest to the given value.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to get index from
    value : float
        Value to find index for
        
    Returns:
    -------
    int
        Index closest to the given value
    """
    array = get_x_array(signal)
    return np.argmin(np.abs(array - value))



def gauss_blur_data(signal, sigma=1):
    """
    Applies Gaussian blur to the data.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to blur
    sigma : float
        Standard deviation for Gaussian kernel
        
    Returns:
    -------
    HyperSpy signal
        Blurred signal
    """
    signal.data = gaussian_filter(signal.data, sigma=sigma)
    return signal


def find_fill_candidate(data, i0, j0):
    """
    Finds the best candidate for filling empty space in the signal.
    
    Parameters:
    ----------
    data : numpy.ndarray
        Signal data
    i0, j0 : int
        Indices of the empty space
        
    Returns:
    -------
    numpy.ndarray
        Candidate signal
    """
    max_y, max_x, _ = data.shape
    
    for i in [i0, i0 - 1, i0 + 1]:
        if i < 0 or i >= max_x:
            continue
        
        for j in [j0, j0 - 1, j0 + 1]:
            if j < 0 or j >= max_y:
                continue
            
            signal = data[j, i, :]
            if np.max(signal) > 0:
                return signal
            
    return data[j0, i0, :]


def fill_empty_spaces(signal):
    """
    Fills all the empty spaces in the signal.
    
    Parameters:
    ----------
    signal : HyperSpy signal
        Signal to fill
        
    Returns:
    -------
    HyperSpy signal
        Filled signal
    """
    s = signal.copy()
    data = s.data
    
    y, x, z = data.shape
    new_data = np.zeros((y, x, z))
    for i in range(x):
        for j in range(y):
            new_data[j, i, :] = find_fill_candidate(data, i, j)
                
    s.data = new_data
    return s
