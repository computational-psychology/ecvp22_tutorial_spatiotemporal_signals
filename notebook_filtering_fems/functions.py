"""
These functions are the basis for the implementation of my edge-sensitivity model.
They are also relevant for quantifying the edge quality of the model.

Last update on 04.06.2022
@author: lynnschmittwilken
"""

import numpy as np


# %%
###############################
#            Drift            #
###############################
def brownian(T: float, pps: float, D: float):
    """Function to create 2d Brownian motion.

    Parameters
    ----------
    T
        Time interval (unit: s)
    pps
        Temporal sampling frequency (unit: Hz)
    D
        Diffusion coefficient (unit: deg**2 / s)

    Returns
    -------
    y
        Displacement array (unit: deg)

    """

    n = int(T*pps)  # Number of drift movements
    dt = 1. / pps   # Time step between two consequent steps (unit: seconds)

    # Generate a 2d stochastic, normally-distributed time series:
    y = np.random.normal(0, 1., [2, n])

    # The average displacement is proportional to dt and D
    y = y * np.sqrt(2.*dt*D)

    # Set initial displacement to 0.
    y = np.insert(y, 0, 0., axis=1)
    return y


def create_drift(T, pps, ppd, D):
    """Function to create 2d drift motion.

    Parameters
    ----------
    T
        Time interval (unit: s)
    pps
        Temporal sampling frequency (unit: Hz)
    ppd
        Spatial resolution (pixels per degree)
    D
        Diffusion coefficient (unit: deg**2 / s)

    Returns
    -------
    y
        Continuous drift array (unit: "continuous px")
    y_int
        Discretized drift array (unit: px)

    """

    # Since our simulations are in px-space, we want to ensure that our drift paths != 0
    cond = 0.
    while (cond == 0.):
        # Generate 2d brownian displacement array
        y = brownian(T, pps, D) * ppd

        # Generate drift path in px from continuous displacement array
        y = np.cumsum(y, axis=-1)
        y_int = np.round(y).astype(int)

        # Sum the horizontal and vertical drift paths in px to make sure that both are != 0:
        cond = y_int[0, :].sum() * y_int[1, :].sum()
    return y, y_int


def apply_drift(stimulus, drift, back_lum=0.5):
    """Create a video in which the stimulus gets shifted over time based on drift.

    Parameters
    ----------
    stimulus
        2D array / image
    drift
        2D discretized drift array (unit: px)
    back_lum
        Intensity value that is used to extend the array

    Returns
    -------
    stimulus_video
        3D array (dimensions: x, y, t) in which the stimulus is shifted over time

    """
    steps = np.size(drift, 1)
    center_x1 = int(np.size(stimulus, 0) / 2)
    center_y1 = int(np.size(stimulus, 1) / 2)

    # Determine the largest displacement and increase stimulus size accordingly
    largest_disp = int(np.abs(drift).max())
    stimulus_extended = np.pad(stimulus, largest_disp, 'constant', constant_values=(back_lum))
    center_x2 = int(np.size(stimulus_extended, 0) / 2)
    center_y2 = int(np.size(stimulus_extended, 1) / 2)

    # Initialize drift video:
    stimulus_video = np.zeros([np.size(stimulus, 0), np.size(stimulus, 1), steps], np.float16)
    stimulus_video[:, :, 0] = stimulus

    for t in range(1, steps):
        x, y = int(drift[0, t]), int(drift[1, t])

        # Create drift video:
        stimulus_video[:, :, t] = stimulus_extended[
                center_x2-center_x1+x:center_x2+center_x1+x,
                center_y2-center_y1+y:center_y2+center_y1+y]
    return stimulus_video.astype(np.float32)
