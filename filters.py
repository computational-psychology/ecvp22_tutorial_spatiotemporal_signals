#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some filter functions
Created on 07.07.2022

@author: lynnschmittwilken
"""

import numpy as np


# %%
###############################
#           Filters           #
###############################
def gauss_fft(fx, fy, sigma: float):
    """Function to create an isotropic Gaussian filter in the frequency domain.

    Parameters
    ----------
    fx
        Array with frequencies in x-direction.
    fy
        Array with frequencies in y-direction.
    sigma
        Sigma that defines the spread of Gaussian filter in deg.

    Returns
    -------
    gauss
        2D Gaussian filter in frequency domain.

    """
    gauss = np.exp(-2. * np.pi**2. * sigma**2. * (fx**2. + fy**2.))
    return gauss


def create_dog(fx, fy, sigma_c, sigma_s):
    """Function to create an isotropic Difference-of-Gaussian filter in the frequency domain

    Parameters
    ----------
    fx
        Array with frequencies in x-direction.
    fy
        Array with frequencies in y-direction.
    sigma_c
        Sigma that defines the spread of the central Gaussian in deg.
    sigma_s
        Sigma that defines the spread of the surround Gaussian in deg.

    Returns
    -------
    dog
        2D Difference-of-Gaussian filter in frequency domain.

    """
    center = gauss_fft(fx, fy, sigma_c)
    surround = gauss_fft(fx, fy, sigma_s)

    dog = center - surround
    return dog


def calculate_peak_freq(dog, fs):
    """Function to calculate peak frequency of a DoG filters defined in frequency space.

    Parameters
    ----------
    dog
        List of 2D Difference-of-Gaussians filters in frequency space.
    fs
        Array with corresponding frequencies.

    Returns
    -------
    peak_freqs
        List of peak frequencies of DoG filters.

    """
    n_filters = len(dog)
    nX = np.size(dog, 0)

    peak_freqs = np.zeros(n_filters)
    for i in range(n_filters):
        filter_row = dog[i][int(nX/2), :]
        max_index = np.where(filter_row == np.max(filter_row))
        max_index = max_index[0][0]
        peak_freqs[i] = np.abs(fs[max_index])
    return peak_freqs


def create_tfilt(tf):
    """Function to create a temporal bandpass filter fitted to the temporal tuning
    properties of macaque simple cells reported in Zheng et al. (2007)

    Parameters
    ----------
    tf
        1d array with temporal frequencies.

    Returns
    -------
    H
        1d temporal bandpass filter.

    """
    # To get a symmetrical filter around 0 Hz, we calculate the absolute tfs:
    tf = np.abs(tf)

    # The equation does not allow tf=0 Hz, so we implement a small workaround and set it to 0
    # manually afterwards
    idx0 = np.where(tf == 0.)[0]
    tf[tf == 0.] = 1.

    # Parameters from fitting the equation to the data of adult macaque V1 cells of Zheng2007:
    m1 = 1.   # 69.3 to actually scale it to the data
    m2 = 22.9
    m3 = 8.1
    m4 = 0.8
    H = m1 * np.exp(-(tf / m2) ** 2.) / (1. + (m3 / tf)**m4)

    if len(idx0):
        H[idx0[0]] = 0.
    return H


def bandpass_filter(fx, fy, fcenter, sigma):
    """Function to create a bandpass filter

    Parameters
    ----------
    fx
        Array with frequencies in x-direction.
    fy
        Array with frequencies in y-direction.
    fcenter
        Center frequency of the bandpass filter
    sigma
        Sigma that defines the spread of the Gaussian in deg.

    Returns
    -------
    bandpass
        2D bandpass filter in frequency domain.

    """
    # Calculate the distance of each 2d spatial frequency from requested center frequency
    distance = np.abs(fcenter - np.sqrt(fx**2. + fy**2.))

    # Create bandpass filter:
    bandpass = 1. / (np.sqrt(2.*np.pi) * sigma) * np.exp(-(distance**2.) / (2.*sigma**2.))
    bandpass = bandpass / bandpass.max()
    return bandpass


# Create spatial RF of retinal ganglion cells from Croner & Kaplan (1995)
def define_retinal_sfilt_fft(fx, fy, rc, Kc, rs, Ks):
    # Spatial response profile in frequency domain
    center = Kc * np.pi * rc**2. * np.exp(-(np.pi*rc*np.sqrt(fx**2.+fy**2.))**2.)
    surround = Ks * np.pi * rs**2. * np.exp(-(np.pi*rs*np.sqrt(fx**2.+fy**2.))**2.)
    sfilt = center - surround
    return sfilt


# Spatiotemporal csf functions taken from Kelly1979:
def st_csf(sfs, tf):
    # The equation does not allow sf=0, so we implement a small workaround
    sfs2 = np.copy(sfs)
    idx = np.where(sfs2 == 0.)
    sfs2[sfs == 0.] = 1.

    # Calculate "velocity" needed for formula
    v = tf / sfs2

    # Calculate contrast sensitivity function:
    k = 6.1 + 7.3 * np.abs(np.log10(v/3.))**3.
    amax = 45.9 / (v + 2.)
    csf = k * v * (2.*np.pi*sfs2)**2. * np.exp((-4.*np.pi*sfs2) / amax)
    csfplt = 1. / csf

    if len(idx):
        csf[idx] = 0.
        csfplt[idx] = 0.
    return csf, csfplt


def create_lowpass(fx, fy, radius, sharpness):
    # Calculate the distance of each frequency from requested frequency
    distance = radius - np.sqrt(fx**2. + fy**2.)
    distance[distance > 0] = 0
    distance = np.abs(distance)

    # Create bandpass filter:
    lowpass = 1. / (np.sqrt(2.*np.pi) * sharpness) * np.exp(-(distance**2.) / (2.*sharpness**2.))
    lowpass = lowpass / lowpass.max()
    return lowpass


def create_loggabor_fft(fx, fy, fo, sigma_fo, angleo, sigma_angleo):
    nY, nX = fx.shape
    fr = np.sqrt(fx**2. + fy**2.)
    fr[int(nY/2), int(nX/2)] = 1.

    # Calculate radial component of the filter:
    radial = np.exp((-(np.log(fr/fo))**2.) / (2. * np.log(sigma_fo)**2.))

    # Undo radius fudge
    radial[int(nY/2), int(nX/2)] = 0.
    fr[int(nY/2), int(nX/2)] = 0.

    # Multiply radial part with lowpass filter to achieve even coverage in corners
    # Lowpass filter will limit the maximum frequency
    lowpass = create_lowpass(fx, fy, radius=fx.max(), sharpness=1.)
    radial = radial * lowpass

    # Calculate angular component of log-Gabor filter
    theta = np.arctan2(fy, fx)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # For each point in the polar-angle-matrix, calculate the angular distance
    # from the filter orientation
    ds = sintheta * np.cos(angleo) - costheta * np.sin(angleo)  # difference in sin
    dc = costheta * np.cos(angleo) + sintheta * np.sin(angleo)  # difference in cos
    dtheta = np.abs(np.arctan2(ds, dc))                         # absolute angular distance
    angular = np.exp((-dtheta**2.) / (2. * sigma_angleo**2.))   # calculate angular filter component

    loggabor = angular * radial
    return loggabor


def create_loggabor(fx, fy, fo, sigma_fo, angleo, sigma_angleo):
    loggabor_fft = create_loggabor_fft(fx, fy, fo, sigma_fo, angleo, sigma_angleo)

    # Get log-Gabor filter in image-space via ifft
    loggabor = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(loggabor_fft)))

    # Real and imaginary part consitute the even- and odd-symmetric filters
    loggabor_even = loggabor.real
    loggabor_odd = np.real(loggabor.imag)
    return loggabor_even, loggabor_odd


# Set up a 2d gaussian based on two sigmas (sigma1, sigma2) and an orientation
def gaussian2d(x, y, sigma, orientation=0):
    # set center to (0, 0)
    center = (0, 0)

    # convert orientation parameter to radians
    theta = np.deg2rad(orientation)

    # determine a, b, c coefficients
    a = (np.cos(theta)**2 / (2*sigma[0]**2)) +\
        (np.sin(theta)**2 / (2*sigma[1]**2))
    b = -(np.sin(2*theta) / (4*sigma[0]**2)) +\
        (np.sin(2*theta) / (4*sigma[1]**2))
    c = (np.sin(theta)**2 / (2*sigma[0]**2)) +\
        (np.cos(theta)**2 / (2*sigma[1]**2))

    # create Gaussian
    gaussian = np.exp(-(a*(x-center[0])**2 +
                      2*b*(x-center[0])*(y-center[1]) +
                      c*(y-center[1])**2))
    return gaussian


# Set uo a dog filter given a sigma for the central and surround Guassians
def define_dog(x, y, sigma_c, sigma_s):
    # Create center and surround anisotropic Gaussian filters
    center = gaussian2d(x, y, sigma=(sigma_c, sigma_c))
    surround = gaussian2d(x, y, sigma=(sigma_s, sigma_s))

    # Normalize each filter by its total sum
    center = center / center.sum()
    surround = surround / surround.sum()

    # Subtract to create differential filter
    dog = center - surround
    return dog


# Temporal RF of M cells adapted from Rucci code:
def M_temporal(tf):
    # parameters reported in the paper by Benardete and Kaplan
    c = .4
    A = 567
    # factor D (in ms. In the original paper it was in s)
    D = 2.2/1000.
    # if Hs = 1 then the integral of the temporal impulse response is 0
    # in the original paper Benardete and Kaplan reported Hs = .98
    # in my simulations we put Hs = 1
    Hs = 1.
    C_12 = 0.056
    T0 = 54.60

    # factor tau_S and tau_L (in seconds. In the original paper it was in ms)
    tau_S = (T0/(1.+(c/C_12)**2.))/1000.
    tau_L = 1.41/1000.
    N_L = 30.30

    # here we compute the impulse response in the Fourier domain
    w2pi = 2.*np.pi*tf
    K = A * np.exp(-1j*w2pi*D) * (1. - Hs/(1. + 1j*w2pi*tau_S)) * ((1./(1.+1j*w2pi*tau_L))**N_L)

    # ... and here we perform the inverse Fourier Transform
    RemSteps = len(K)-2
    FinK = np.append(K, np.conj(np.flip(K[-RemSteps:-1])))
    M_T = np.real(np.fft.ifft(FinK))
    return K, M_T


# Temporal CSF as defined in Watson (1986) with parameters fitted to Robson (1966) data:
def csf_temporal(tf):
    tau1 = 6.22/1000.
    kappa = 1.33
    tau2 = tau1 / kappa
    n1 = 9.
    n2 = 10.
    xi = 214.
    zeta = 0.9

    # here we compute the impulse response in the Fourier domain
    w2pi = 2.*np.pi*tf
    H1 = (-1j*w2pi*tau1 + 1.) ** (-n1)
    H2 = (-1j*w2pi*tau2 + 1.) ** (-n2)
    H = xi * (H1 - zeta*H2)
    return H


# Temporal filter based on LGN data from Derrington (1984), slightly adapted
def LGN_temporal(tf):
    # Necessary to get a symmetrical filter around 0 Hz:
    tf = np.abs(tf)

    # Parameters for linear magnocellular unit
    S1 = 137.
    k1 = 0.054
    S2 = 138.
    k2 = 0.156
    H = S1 * np.exp(-k1*tf) - S2 * np.exp(-k2*tf)

    # Neccessary because f(0) = -1
    H[H < 0] = 0.
    return H


def V1_temporal(tf):
    # To get a symmetrical filter around 0 Hz, we calculate the absolute tfs:
    tf = np.abs(tf)

    # The equation does not allow tf=0 Hz, so we implement a small workaround and set it to 0
    # manually afterwards
    idx0 = np.where(tf == 0.)[0]
    tf[tf == 0.] = 1.

    # Parameters from fitting the equation to the data of adult macaque V1 cells of Zheng2007:
    m1 = 1.   # 69.3 to actually scale it to the data
    m2 = 22.9
    m3 = 8.1
    m4 = 0.8
    H = m1 * np.exp(-(tf / m2) ** 2.) / (1. + (m3 / tf)**m4)

    if len(idx0):
        H[idx0[0]] = 0.
    return H
