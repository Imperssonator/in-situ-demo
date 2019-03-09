#!/usr/bin/env python

# Basic
import numpy as np
from scipy.signal import medfilt
import glob
import os
import pandas as pd
from matplotlib import pyplot as plt

# pyFAI
import pyFAI
import pygix
from pygix import plotting as ppl
import fabio

data_dir = "/Users/nils/CC/CMS Data/Nils/insitu_air"
calib_csv = os.path.join(data_dir, 'calib.csv')

def parse_filename(file):
    
    parts = os.path.basename(file).split('_')
    parsed = pd.Series()
    
    try:
        # Sample name is the first few things up to -4
        parsed['sample'] = '_'.join(parts[0:-5])
        
        # The rest is just acquisition parameters
        parsed['t'] = float(parts[-5][:-1])
        parsed['theta'] = float(parts[-4][2:])
        parsed['exp_time'] = float(parts[-3][:-1])
        parsed['stamp'] = int(parts[-2])
        parsed['mode'] = parts[-1].split('.')[0]
        
    except:
        # Sample name is the first few things up to -4
        parsed['sample'] = '_'.join(parts[0:-6])
        
        # The rest is just acquisition parameters
        parsed['t'] = float(parts[-6][:-1])
        parsed['theta'] = float(parts[-5][2:])
        parsed['exp_time'] = float(parts[-4][:-1])
        parsed['stamp'] = int(parts[-3])
        parsed['burst'] = int(parts[-2])
        parsed['mode'] = parts[-1].split('.')[0]

        
    return parsed


def build_master_table(raw_files, im_shape_pix=(195,487)):

    # Build a DataFrame from the list of raw file names
    df = pd.DataFrame()
    df['tiff'] = pd.Series(raw_files)
    df = pd.concat([df, df['tiff'].apply(parse_filename)],
                          axis=1)

    return df


def setup_detector(theta, calib_csv=calib_csv):
    
    det_param_df = pd.read_csv(calib_csv)
    det_kwargs = dict(zip(det_param_df['param'], det_param_df['val']))
    
    # Detector only holds pixel size for whatever reason
    detector = pyFAI.detectors.Detector(det_kwargs['det_pix'], det_kwargs['det_pix'])
    
    # Transform holds everything else
    pg = pygix.Transform(incident_angle = theta, detector=detector,
                         **{k:v for k,v in det_kwargs.items() if k != 'det_pix'})

    return pg
    
    
def get_data(df, sample):
    
    # Function to automatically load, reshape, offset, and median filter raw .tiff data
    
#     data = fabio.open(df['tiff'].loc[sample]).data.reshape(det_shape) - offset
    data = fabio.open(df['tiff'].loc[sample]).data
    return data
#     return data


def get_blank(dfb, dfw, sample, blank_set='si_pos1'):
    """ Given a sample row from a regular df, find blank with corresponding theta and
    scale for exposure time"""
    
    dfb_filter = dfb.loc[dfb['sample']==blank_set].loc[dfb['mode']==dfw['mode'].loc[sample]]
    dfb_sorted = dfb_filter.iloc[(dfb_filter['theta']-dfw['theta'].loc[sample]).abs().argsort()]
    blank_match = dfb_sorted.index.values[0]
    blank_raw = get_data(dfb_sorted, blank_match)
    blank_scaled = blank_raw * dfw['exp_time'].loc[sample] / dfb['exp_time'].loc[blank_match]

    return blank_scaled


def get_sector(df, sample, chi, dark=None,
               chi_width=4, radial_range=(0,2)):
    
    data = get_data(df, sample)
    theta = df['theta'].loc[sample]
    pg = setup_detector(theta)
    
    ii, q = pg.profile_sector(data, npt=1000, chi_pos=chi,
                              chi_width=chi_width, radial_range=radial_range,
                              correctSolidAngle=True, dark=dark,
                              method='lut', unit='q_A^-1')
    ii[ii<1]=1
    
    return ii, q


def get_ip_box(df, sample, op_pos, dark=None,
               op_width=0.05, ip_range=(-2,0.1)):
    
    data = get_data(df, sample)
    theta = df['theta'].loc[sample]
    pg = setup_detector(theta)
    
    ii, q = pg.profile_ip_box(data, npt=1000, op_pos=op_pos,
                             op_width=op_width, ip_range=ip_range,
                             correctSolidAngle=True, dark=dark,
                             method='lut', unit='q_A^-1')
    
    ii[ii<1]=1
    
    return np.flip(ii), np.flip(-q)


def get_pole_figure(df, sample, dark=None, chi_range=(-90,0), q_range=(0,2), npt=(180,180)):
    
    data = get_data(df, sample)
    theta = df['theta'].loc[sample]
    pg = setup_detector(theta)
    
    intensity, q_abs, chi = pg.transform_polar(data, unit='A', npt=npt,
                                               chi_range=chi_range, q_range=q_range,
                                               correctSolidAngle=True, dark=dark,
                                               method='splitpix')
    intensity[intensity<1]=1
    return intensity, q_abs, chi


def bg_corr_slice(A, x, xb, xbg):
    """
    Given 2D array with x scale (for the columns), get
    an integrated slice within the x values xb, with
    background subtracted from the x locations in xbg
    (typically two locations just outside of xb).
    3 lines will be sampled at each xbg (+/- 1 pixel)
    
    A: array, shape[1]=len(x)
    x: 1-D array, scale for column positions of A (like an x-axis)
    xb: tuple (lb, ub)
    xbg: list-like of individual x locations to sample
    
    Return: cut (bg-corrected slice of A)
    """

    x_cols = np.array([np.where(x>xb[0])[0][0], np.where(x>xb[1])[0][0]])
    x_bg_cols = np.array([np.where(x>val)[0][0] for val in xbg])
    x_bg_cols = np.hstack([x_bg_cols, x_bg_cols+1, x_bg_cols-1])

    A_cut = A[:,x_cols[0]:x_cols[1]]
    bg_cut = np.tile(np.mean(A[:,x_bg_cols], axis=1), (A_cut.shape[1], 1)).T
    cut = np.trapz(A_cut-bg_cut, x[x_cols[0]:x_cols[1]])
    
    return cut


def Hermans(ii, chi):
    """
    This function will sin-weight the intensity... do not pass sin-weighted intensity...
    """
    sin_chi = np.sin(np.deg2rad(chi))
    cos2chi = np.cos(np.deg2rad(chi)) ** 2
    expect = np.sum(ii * cos2chi * sin_chi) / np.sum(ii * sin_chi)
    return (3*expect-1)/2


def interp_nans(data):
    "Fill NaNs in an array with linear interpolation"
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    return data


def show_sample(df,sample, newfig=True, dark=None, log=True, prcs=(2,99.8), figsize=(5,5)):
    
    # Inputs:
    # df: a dataframe with parsed filenames
    # sample: an integer corresponding to the row of the dataframe you'd like to analyze
    # newfig false if you're using this as part of a subplot
    
    # setup_detector defines a transform with the parameters we got from calibration
    pg = setup_detector(df['theta'].loc[sample])
    
    # Generate the corrected image
    # get_data loads the tif file and 3x3 median-filters it (de-zinger)
    imgt, qxy, qz = pg.transform_reciprocal(get_data(df,sample),
                                            method='lut',
                                            correctSolidAngle=True,
                                            unit='A', dark=dark)
    
    imgt[imgt<1]=1
    if log:
        imgt = np.log(imgt)
    
    # Calculate where the color scale should be bounded
    clim = np.percentile(imgt, prcs)
    
    if newfig:
        plt.figure(figsize=figsize)
    
    # pygix built-in plotting with nice formatting
    ppl.implot(imgt, -qxy, qz, mode='rsma',
               cmap="terrain", clim=clim,
               xlim=(-0.1,2), ylim=(-0.1,2),
               newfig=False, show=False)
               
               
def pcolor(img, newfig=True, figsize=(5,4), extent=None,
           clip_val=None, prcs=(2,99.5), log=False,
           cmap='terrain', aspect='auto', origin=None):
    
    if newfig:
        plt.figure(figsize=figsize)
        
    if log:
        img = np.log(img)
    
    if clip_val:
        cmin, cmax = np.percentile(img[img>clip_val], prcs)
    else:
        cmin, cmax = np.percentile(img, prcs)
        
    plt.imshow(img, cmap=cmap, vmin=cmin, vmax=cmax,
               extent=extent, aspect=aspect, origin=origin)