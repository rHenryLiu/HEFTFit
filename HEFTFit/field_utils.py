import argparse
import gc
import os
from pathlib import Path
import warnings

import asdf
import numpy as np
import yaml
import numba
#from np.fft import fftfreq, fftn, ifftn
from scipy.fft import rfftn, irfftn

from abacusnbody.metadata import get_meta

from asdf.exceptions import AsdfWarning
warnings.filterwarnings('ignore', category=AsdfWarning)

# DEFAULTS = {'path2config': 'config/abacus_heft.yaml'}

def compress_asdf(asdf_fn, table, header):
    r"""
    Compress the dictionaries `table` and `header` using blsc into an ASDF file, `asdf_fn`.
    """
    # cram into a dictionary
    data_dict = {}
    for field in table.keys():
        data_dict[field] = table[field]

    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }

    # set compression options here
    compression_kwargs=dict(typesize="auto", shuffle="shuffle", compression_block_size=12*1024**2, blosc_block_size=3*1024**2, nthreads=4)
    with asdf.AsdfFile(data_tree) as af, open(asdf_fn, 'wb') as fp: # where data_tree is the ASDF dict tree structure
        af.write_to(fp)#, all_array_compression='blsc', compression_kwargs=compression_kwargs)

def load_lagrangians(path):
    """
    Load initial (Lagrangian) positions for the MillenniumTNG Simulation
    """
    lagrangians = []
    for i in range(10):
        i = i+1
        # lagrangians.append(np.load(path + 'lagrangian_position_sorted_264_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
        lagrangians.append(np.load(path + 'position_sorted_000_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
    return np.concatenate(lagrangians)

def load_positions(z, path):
    """
    Load positions for the MillenniumTNG Simulation
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    pos = []
    for i in range(10):
        i = i+1
        pos.append(np.load(path + 'position_sorted_' + string + '_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
    return np.concatenate(pos)

def load_velocities(z, path):
    """
    Load velocities for the MillenniumTNG Simulation
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    vel = []
    for i in range(10):
        i = i+1
        vel.append(np.load(path + 'velocity_sorted_' + string + '_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
    return np.concatenate(vel)

def load_tau(z, path):
    """
    Load velocities for the MillenniumTNG Simulation
    """
    if z==0.:
        string = '264'
    elif z==0.5:
        string = '214'
    elif z==1.0:
        string = '179'
    else:
        raise Exception("Redshift z is not one of the allowed values")
    
    result = np.load(path + 'tau_3d_snap_' + string + '.npy')
    return result



def gaussian_filter(field, nmesh, lbox, kcut):
    """
    Apply a fourier space gaussian filter to a field.

    Parameters
    ---------
    field : array_like
        the field to filter.
    nmesh : int
        size of the mesh.
    lbox : float
        size of the box.
    kcut : float
        the exponential cutoff to use in the gaussian filter

    Returns
    -------
    f_filt : array_like
        Gaussian filtered version of field
    """

    # fourier transform field
    field_fft = rfftn(field, workers=-1).astype(np.complex64)

    # inverse fourier transform
    f_filt = irfftn(filter_field(field_fft, nmesh, lbox, kcut), workers=-1).astype(np.float32)
    return f_filt

@numba.njit(parallel=True, fastmath=True)
def filter_field(delta_k, n1d, L, kcut, dtype=np.float32):

    r"""
    Compute nabla^2 delta in Fourier space.

    Parameters
    ----------
    delta_k : array_like
        Fourier 3D field.
    n1d : int
        size of the 3d array along x and y dimension.
    L : float
        box size of the simulation.
    kcut : float
        smoothing scale in Fourier space.
    dtype : np.dtype
        float type (32 or 64) to use in calculations.

    Returns
    -------
    n2_fft : array_like
        Fourier 3D field.
    """
    # define number of modes along last dimension
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    dk = dtype(2. * np.pi / L)
    norm = dtype(2. * kcut**2)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        kx = dtype(i)*dk if i < n1d//2 else dtype(i - n1d)*dk
        for j in range(n1d):
            ky = dtype(j)*dk if j < n1d//2 else dtype(j - n1d)*dk
            for k in range(kzlen):
                kz = dtype(k)*dk
                kmag2 = (kx**2 + ky**2 + kz**2)
                delta_k[i, j, k] = np.exp(-kmag2 / norm) * delta_k[i, j, k]
    return delta_k


@numba.njit(parallel=True, fastmath=True)
def get_n2_fft(delta_k, n1d, L, dtype=np.float32):
    r"""
    Compute nabla^2 delta in Fourier space.

    Parameters
    ----------
    delta_k : array_like
        Fourier 3D field.
    n1d : int
        size of the 3d array along x and y dimension.
    L : float
        box size of the simulation.
    dtype : np.dtype
        float type (32 or 64) to use in calculations.

    Returns
    -------
    n2_fft : array_like
        Fourier 3D field.
    """
    # define number of modes along last dimension
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    dk = dtype(2. * np.pi / L)

    # initialize field
    n2_fft = np.zeros((n1d, n1d, kzlen), dtype=delta_k.dtype)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        kx = dtype(i)*dk if i < n1d//2 else dtype(i - n1d)*dk
        for j in range(n1d):
            ky = dtype(j)*dk if j < n1d//2 else dtype(j - n1d)*dk
            for k in range(kzlen):
                kz = dtype(k)*dk
                kmag2 = (kx**2 + ky**2 + kz**2)
                n2_fft[i, j, k] = -kmag2 * delta_k[i, j, k]
    return n2_fft

@numba.njit(parallel=True, fastmath=True)
def get_sij_fft(i_comp, j_comp, delta_k, n1d, L, dtype=np.float32):
    r"""
    Compute ijth component of the tidal tensor in Fourier space.

    Parameters
    ----------
    i_comp : int
        ith component of the tensor.
    j_comp : int
        jth component of the tensor.
    delta_k : array_like
        Fourier 3D field.
    n1d : int
        size of the 3d array along x and y dimension.
    L : float
        box size of the simulation.
    dtype : np.dtype
        float type (32 or 64) to use in calculations.

    Returns
    -------
    s_ij_fft : array_like
        Fourier 3D field.
    """
    # define number of modes along last dimension
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    dk = dtype(2. * np.pi / L)
    if i_comp == j_comp:
        delta_ij_over_3 = dtype(1./3.)
    else:
        delta_ij_over_3 = dtype(0.)

    # initialize field
    s_ij_fft = np.zeros((n1d, n1d, kzlen), dtype=delta_k.dtype)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        kx = dtype(i)*dk if i < n1d//2 else dtype(i - n1d)*dk
        if i_comp == 0:
            ki = kx
        if j_comp == 0:
            kj = kx
        for j in range(n1d):
            ky = dtype(j)*dk if j < n1d//2 else dtype(j - n1d)*dk
            if i_comp == 1:
                ki = ky
            if j_comp == 1:
                kj = ky
            for k in range(kzlen):
                kz = dtype(k)*dk
                if i + j + k > 0:
                    kmag2_inv = dtype(1.)/(kx**2 + ky**2 + kz**2)
                else:
                    kmag2_inv = dtype(0.)
                if i_comp == 2:
                    ki = kz
                if j_comp == 2:
                    kj = kz
                s_ij_fft[i, j, k] = delta_k[i, j, k] * (ki *kj * kmag2_inv - delta_ij_over_3)
    return s_ij_fft

@numba.njit(parallel=True, fastmath=True)
def add_ij(final_field, field_to_add, n1d, factor=1., dtype=np.float32):
    r"""
    Add field `field_to_add` to `final_field` with a constant factor.
    """
    factor = dtype(factor)
    for i in numba.prange(n1d):
        for j in range(n1d):
            for k in range(n1d):
                final_field[i, j, k] += factor * field_to_add[i, j, k]**2
    return

def get_dk_to_s2(delta_k, nmesh, lbox):
    r"""
    Computes the square tidal field from the density FFT `s^2 = s_ij s_ij`,
    where `s_ij = (k_i k_j / k^2 - delta_ij / 3 ) * delta_k`.

    Parameters
    ----------
    delta_k : array_like
        Fourier transformed density field.
    nmesh : int
        size of the mesh.
    lbox : float
        size of the box.

    Returns
    -------
    tidesq :
        the tidal field (s^2).
    """
    # Compute the symmetric tide at every Fourier mode which we'll reshape later
    # Order is xx, xy, xz, yy, yz, zz
    jvec = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    # compute s_ij and do the summation
    tidesq = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
    for i in range(len(jvec)):
        if jvec[i][0] != jvec[i][1]:
            factor = 2.
        else:
            factor = 1.
        add_ij(tidesq, irfftn(get_sij_fft(jvec[i][0], jvec[i][1], delta_k, nmesh, lbox), workers=-1), nmesh, factor)
    return tidesq

def get_dk_to_n2(delta_k, nmesh, lbox):
    """
    Computes the density curvature from the density field: nabla^2 delta = IFFT(-k^2 delta_k)
    Parameters
    ----------
    delta_k : array_like
        Fourier transformed density field.
    nmesh : int
        size of the mesh.
    lbox : float
        size of the box.

    Returns
    -------
    real_gradsqdelta : array_like
        the nabla^2 delta field
    """
    # Compute -k^2 delta which is the gradient
    nabla2delta = irfftn(get_n2_fft(delta_k, nmesh, lbox), workers=-1).astype(np.float32)
    return nabla2delta

def get_fields(delta_lin, Lbox, nmesh):
    """
    Return the fields delta, delta^2, s^2, nabla^2 given the linear density field.
    """

    # get delta
    delta_fft = rfftn(delta_lin, workers=-1).astype(np.complex64)
    fmean = np.mean(delta_lin, dtype=np.float64)
    d = delta_lin-fmean
    gc.collect()
    print("Generated delta")

    # get delta^2
    d2 = delta_lin * delta_lin
    fmean = np.mean(d2, dtype=np.float64)
    d2 -= fmean
    del delta_lin
    gc.collect()
    print("Generated delta^2")

    # get s^2
    s2 = get_dk_to_s2(delta_fft, nmesh, Lbox)
    fmean = np.mean(s2, dtype=np.float64)
    s2 -= fmean
    print("Generated s_ij s^ij")

    # get n^2
    n2 = get_dk_to_n2(delta_fft, nmesh, Lbox)
    print("Generated nabla^2")

    return d, d2, s2, n2