"""
Some helper functions to keep the Jupyter notebook clean

@author: lynnschmittwilken
Created on 24.06.2021
"""

import numpy as np
import matplotlib.pyplot as plt


# Helper function for radial averaging:
def radial_mean(x, radii, rounding_factor=0.25, only_full_circles=False):
    """
    Radial mean of a 2D array. Values with equal (rounded) radii will be assigned to bins, of
    which then the mean will be calculated.
    parameters
    ------------
    x                  - 2D or 3D array of values to be averaged
    radii              - same shape as x containing the radius of each pixel, over which to average
    rounding_factor    - determines the granularity of the bins
    only_full_circles  - exclude radius bins greater than the maximal width/height radius

    returns
    ---------
    means              - 1D array; mean of each radius bin
    bins               - 1D array; radius corresponding to each bin
    """

    if x.ndim == 2:
        x = np.expand_dims(x, -1)

    # create bins by rounding to int
    radii = (radii / rounding_factor).astype(int)

    # Initiate output variable:
    nrep = x.shape[-1]
    means = np.zeros([radii.max()+1, nrep])

    for i in range(nrep):
        # creates bins from 0 to radii.max(); for all i, adds x[i] to sum_bins[radii[i]]
        sum_bins = np.bincount(radii.ravel(), x[:, :, i].ravel())

        # count how often a radius is represented
        count_bins = np.bincount(radii.ravel())

        # mean of each radius bin
        means[:, i] = np.ma.divide(sum_bins, count_bins)

    # radius each bin represents
    bins = np.arange(0, means.shape[0]) * rounding_factor

    if only_full_circles:
        max_bin = min(radii[0, :].min(), radii[:, 0].min(), radii[-1, :].min(), radii[:, -1].min())
        means = means[:max_bin, :]
        bins = bins[:max_bin]
    return means, bins


###############################
#          Plotting           #
###############################
def plot_grating(grating_2d, gpower_2d, grating_1d, gpower_1d, extent, x, sff):
    sff_ext = (sff[0], sff[-1], sff[0], sff[-1])

    plt.figure(figsize=(18, 3))
    plt.subplot(141)
    plt.imshow(grating_2d, cmap='gray', extent=extent+extent)
    plt.colorbar()
    plt.title('2d grating')
    plt.xlabel('deg')
    plt.ylabel('deg')

    plt.subplot(142)
    plt.imshow(np.log10(gpower_2d + 0.001), extent=sff_ext, vmax=-2)
    plt.colorbar()
    plt.title('Power spectrum (log)')
    plt.xlabel('cpd')
    plt.ylabel('cpd')

    plt.subplot(143)
    plt.plot(x, grating_1d)
    plt.title('1d grating')
    plt.xlabel('deg')
    plt.ylabel('Luminance')

    plt.subplot(144)
    plt.plot(sff, np.log10(gpower_1d+0.001))
    plt.xlabel('cpd')
    plt.ylabel('Power')
    plt.show()


def plot_drift(grating_2d, extent, path, ppd, t, gdrift, C, gd_power, tff):
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.imshow(grating_2d, cmap='gray', extent=extent+extent)
    plt.plot(path[0, :] / ppd, path[1, :] / ppd, 'r')
    plt.title('Grating + drift')
    plt.xlabel('cpd')
    plt.ylabel('cpd')

    plt.subplot(132)
    plt.plot(t, gdrift, 'c.')
    plt.plot(t, gdrift, alpha=0.2)
    plt.ylim(-C*1.2, C*1.2)
    plt.title('Luminance over time')
    plt.xlabel('Time in s')
    plt.ylabel('Luminance')

    plt.subplot(133)
    plt.plot(tff, np.log10(gd_power), 'c.')
    plt.plot(tff, np.log10(gd_power), alpha=0.2)
    plt.ylim(-3., 6.)
    plt.xlim(0.1, 50.)
    plt.xscale('log')
    plt.xticks([0.1, 1, 10, 50], [0.1, 1, 10, 50])
    plt.title('Temporal power')
    plt.xlabel('tf in Hz')
    plt.ylabel('Power in dB')
    plt.show()


def plot_results1(outputs, sff, tff):
    # We only need the output for positive temporal frequencies:
    tff_cut = tff[int(len(tff)/2)+1::]
    outputs_cut = outputs[int(len(tff)/2)+1::, :]

    # In most papers of the Rucci group, max power values in db are equal to zero.
    # I am assuming that they divide by the global maximum:
    outputs_db = 10. * np.log10(outputs_cut / outputs_cut.max())

    plt.figure(figsize=(22, 4))
    plt.subplot(141, aspect=0.8)
    plt.pcolormesh(sff, tff_cut, outputs_db, cmap='hot', vmin=-60, vmax=0., shading='auto')
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar()
    plt.ylim(0.5, 50.)
    plt.xlim(0.5, 30.)
    plt.yticks([1, 10, 50], [1, 10, 50])
    plt.xticks([1, 10, 30], [1, 10, 30])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('tfs (Hz)')

    # Find indices where sf is 1 cpd and 8 cpd
    idx_1cpd = np.abs(sff-1).argmin()
    idx_8cpd = np.abs(sff-8).argmin()

    plt.subplot(142)
    plt.plot(tff_cut, outputs_db[:, idx_1cpd], 'r.',
             label=str(np.round(sff[idx_1cpd])) + ' cpd')
    plt.plot(tff_cut, outputs_db[:, idx_1cpd], 'r', alpha=0.2)
    plt.plot(tff_cut, outputs_db[:, idx_8cpd], 'g.',
             label=str(np.round(sff[idx_8cpd])) + ' cpd')
    plt.plot(tff_cut, outputs_db[:, idx_8cpd], 'g', alpha=0.2)
    plt.xscale('log')
    plt.ylim(-60., 0.)
    plt.xlim(0.5, 80.)
    plt.xticks([1, 10, 50], [1, 10, 50])
    plt.xlabel('tfs (Hz)')
    plt.ylabel('Temporal power (dB)')
    plt.legend()

    # Find indices where tf is 10 Hz
    idx_10hz = np.abs(tff_cut-10).argmin()

    plt.subplot(143)
    plt.plot(sff, outputs_db[idx_10hz, :], 'C0.',
             label=str(np.round(tff_cut[idx_10hz], 1)) + ' Hz')
    plt.plot(sff, outputs_db[idx_10hz, :], 'C0', alpha=0.2)
    plt.xscale('log')
    plt.ylim(-60)
    plt.xlim(0.1, 30.)
    plt.xticks([0.1, 1, 10], [0.1, 1, 10])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('Spectral power (dB)')
    plt.legend()

    plt.subplot(144)
    plt.plot(sff, outputs_db.mean(0), 'C0.')
    plt.plot(sff, outputs_db.mean(0), 'C0', alpha=0.2)
    plt.xscale('log')
    plt.ylim(-45, -15)
    plt.xlim(0.9, 30.)
    plt.xticks([1, 10, 30], [1, 10, 30])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('Temporal power (dB)')
    plt.show()


def plot_results2(outputs, sff, tff):
    # We only need the output for positive temporal frequencies:
    sff_cut = sff[int(len(sff)/2)::]
    tff_cut = tff[int(len(tff)/2)::]
    outputs_cut = outputs[int(len(tff)/2)::, int(len(sff)/2)::]

    # Decrease power for increasing sfs according to natural image statistics.
    # Should be sfs**1.9 but sfs**1.8 leads to a better fit with Kuang2012.
    outputs_cut1 = outputs_cut / outputs_cut.max()
#    outputs_cut2 = outputs_cut / outputs_cut.max() / (sff_1d**3.)
#    outputs_cut3 = outputs_cut / outputs_cut.max() / (sff_1d**4.)
#    outputs_cut4 = outputs_cut / outputs_cut.max() / (sff_1d**5.)

    # Calculate in dB
    outputs_db = 10. * np.log10(outputs_cut1)
#    outputs_db2 = 10. * np.log10(outputs_cut2)
#    outputs_db3 = 10. * np.log10(outputs_cut3)
#    outputs_db4 = 10. * np.log10(outputs_cut4)

    plt.figure(figsize=(22, 4))
    plt.subplot(141, aspect=1.2)
    plt.pcolormesh(sff_cut, tff_cut, outputs_db, cmap='hot', vmin=-75, vmax=-45, shading='auto')
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar()
    plt.ylim(4., 80.)
    plt.xlim(0.4, 20.)
    plt.yticks([10, 30, 80], [10, 30, 80])
    plt.xticks([1, 10], [1, 10])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('tfs (Hz)')

    # Find indices where sf is 0.5 cpd
    idx_05cpd = np.abs(sff_cut-0.5).argmin()

    plt.subplot(142)
    plt.plot(tff_cut, outputs_db[:, idx_05cpd], 'C0.',
             label=str(np.round(sff_cut[idx_05cpd])) + ' cpd')
    plt.xscale('log')
    plt.ylim(-80, -35)
    plt.xlim(4, 50.)
    plt.xticks([10, 50], [10, 50])
    plt.xlabel('tfs (Hz)')
    plt.ylabel('Spectral power (dB)')
    plt.legend()

    # Find indices where tf is 4, 8, and 16 Hz
    idx_4hz = np.abs(tff_cut-4).argmin()
    idx_8hz = np.abs(tff_cut-8).argmin()
    idx_16hz = np.abs(tff_cut-16).argmin()

    plt.subplot(143)
    plt.plot(sff_cut, outputs_db[idx_4hz, :], 'darkblue', marker='.',
             label=str(np.round(tff_cut[idx_4hz], 1)) + ' Hz')
    plt.plot(sff_cut, outputs_db[idx_8hz, :], 'royalblue', marker='.',
             label=str(np.round(tff_cut[idx_8hz], 1)) + ' Hz')
    plt.plot(sff_cut, outputs_db[idx_16hz, :], 'lightseagreen', marker='.',
             label=str(np.round(tff_cut[idx_16hz], 1)) + ' Hz')
    plt.xscale('log')
    plt.ylim(-75, -20)
    plt.xlim(0.5, 25.)
    plt.xticks([1, 10], [1, 10])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('Spectral power (dB)')
    plt.legend()

    plt.subplot(144)
    plt.plot(sff_cut, outputs_db[1::, :].mean(0), 'r.', label='k**-1.9')
#    plt.plot(sff_1d, outputs_db2[1::, :].mean(0), 'darkblue', marker='.', label='k**-3.')
#    plt.plot(sff_1d, outputs_db3[1::, :].mean(0), 'royalblue', marker='.', label='k**-4.')
#    plt.plot(sff_1d, outputs_db4[1::, :].mean(0), 'lightseagreen', marker='.', label='k**-5.')
    plt.xscale('log')
#    plt.ylim(-75, -20)
    plt.xlim(0.95, 30.)
    plt.xticks([1, 10], [1, 10])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('Temporal power (dB)')
    plt.legend()
    plt.show()


def plot_results3(outputs, outputs_rot, sff, tff):
    # Let's get rid of negative frequencies:
    nT = len(tff)
    tff = tff[int(nT/2.)::]

    # There can be negative sffs or not. Let's treat these cases seperately:
    if any(sff < 0):
        nX = len(sff)
        sff = sff[int(nX/2.)::]
        outputs = outputs[int(nX/2.)::, int(nT/2.)::] / outputs.max()
        outputs_rot = outputs_rot[int(nX/2.)::, int(nT/2.)::] / outputs_rot.max()
        prefix = 'Simple average'
    else:
        outputs = outputs[:, int(nT/2.)::] / outputs.max()
        outputs_rot = outputs_rot[:, int(nT/2.)::] / outputs_rot.max()
        prefix = 'Radial average'

    # Transpose and log
    outputs_db = 10. * np.log10(np.transpose(outputs))
    outputs_rot_db = 10. * np.log10(np.transpose(outputs_rot))

    plt.figure(figsize=(18, 4))
    plt.subplot(121, aspect=0.8)
    plt.pcolormesh(sff, tff[1::], outputs_db[1::], cmap='hot', shading='auto')
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar()
    plt.ylim(0.5, 50.)
    plt.xlim(0.5, 30.)
    plt.yticks([1, 10, 50], [1, 10, 50])
    plt.xticks([1, 10, 30], [1, 10, 30])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('tfs (Hz)')
    plt.title(prefix)

    plt.subplot(122, aspect=0.8)
    plt.pcolormesh(sff, tff[1::], outputs_rot_db[1::], cmap='hot', shading='auto')
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar()
    plt.ylim(0.5, 50.)
    plt.xlim(0.5, 30.)
    plt.yticks([1, 10, 50], [1, 10, 50])
    plt.xticks([1, 10, 30], [1, 10, 30])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('tfs (Hz)')
    plt.title(prefix)
    plt.show()


def plot_results4(outputs, outputs_filt, sff, tff):
    # Let's get rid of negative frequencies:
    nT = len(tff)
    tff = tff[int(nT/2.)::]

    # There can be negative sffs or not. Let's treat these cases seperately:
    if any(sff < 0):
        nX = len(sff)
        sff = sff[int(nX/2.)::]
        outputs = outputs[int(nX/2.)::, int(nT/2.)::] / outputs.max()
        outputs_filt = outputs_filt[int(nX/2.)::, int(nT/2.)::] / outputs_filt.max()
    else:
        outputs = outputs[:, int(nT/2.)::] / outputs.max()
        outputs_filt = outputs_filt[:, int(nT/2.)::] / outputs_filt.max()

    # Transpose and log
    outputs_db = 10. * np.log10(np.transpose(outputs))
    outputs_filt_db = 10. * np.log10(np.transpose(outputs_filt))

    plt.figure(figsize=(18, 4))
    plt.subplot(121, aspect=0.8)
    plt.pcolormesh(sff, tff[1::], outputs_db[1::], cmap='hot', shading='auto')
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar()
    plt.ylim(0.5, 50.)
    plt.xlim(0.5, 30.)
    plt.yticks([1, 10, 50], [1, 10, 50])
    plt.xticks([1, 10, 30], [1, 10, 30])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('tfs (Hz)')
    plt.title('Not filtered')

    plt.subplot(122, aspect=0.8)
    plt.pcolormesh(sff, tff[1::], outputs_filt_db[1::], cmap='hot', shading='auto')
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar()
    plt.ylim(0.5, 50.)
    plt.xlim(0.5, 30.)
    plt.yticks([1, 10, 50], [1, 10, 50])
    plt.xticks([1, 10, 30], [1, 10, 30])
    plt.xlabel('sfs (cpd)')
    plt.ylabel('tfs (Hz)')
    plt.title('Filtered')
    plt.show()
