import argparse
import gc
import os
from pathlib import Path
import warnings
import time
import sys

import sys
import asdf
import numpy as np
import matplotlib.pyplot as plt
import yaml
import numba
from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq
from scipy.optimize import minimize
import scipy as sp

from classy import Class
import abacusnbody
import abacusnbody.analysis
from abacusnbody.analysis.tsc import tsc_parallel #put it on a grid using tsc interpolation
from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak #computes power spectrum from density contrast, not specific to abacus 
# from abacusnbody.analysis.power_spectrum import index_3d_rfft

# from obtain_IC_fields import *

from asdf.exceptions import AsdfWarning
warnings.filterwarnings('ignore', category=AsdfWarning)

sys.path.append('../velocileptors') # clone the velocileptors from github 
from velocileptors.LPT.cleft_fftw import CLEFT
sys.path.append('../') # Add the HEFTFit package
from HEFTFit.field_utils import *
from HEFTFit.HEFTFit import HEFTFit


t0 = time.time()
# Configs
print('Load Configs', time.time()-t0)
heft_dir = '/pscratch/sd/r/rhliu/projects/heft_scratch/'
nmesh = 1080
kcut = 0.
z_mock = 0. # config['sim_params']['z_mock']
# z_mock = 1. # config['sim_params']['z_mock']

sim_name = 'MTNG'
save_dir = Path(heft_dir) / sim_name
pcle_dir = save_dir / "pcles"

paste = "TSC"
pcle_type = "A"
factors_fields = {'delta': 1., 'delta2': 2., 'nabla2': 1., 'tidal2': 2}

nmesh = 1080
Lbox = 500
z_ic = 63.
h = 67.76/100

# Making growth factor for MTNG
print('Making growth factor for MTNG', time.time()-t0)
cosmo = {}
cosmo['output'] = 'mPk mTk'
cosmo['P_k_max_h/Mpc'] = 20.

cosmo['H0'] = h*100
cosmo['omega_b'] = 0.0486 * (h)**2
cosmo['omega_cdm'] = (0.3089 - 0.0486) * (h)**2
cosmo['Omega_Lambda'] = 0.6911
cosmo['z_max_pk'] = 10.0


pkclass = Class()
pkclass.set(cosmo)
pkclass.compute()

D_mock = pkclass.scale_independent_growth_factor(z_mock)
D_ic = pkclass.scale_independent_growth_factor(z_ic)
D_ratio = D_mock/D_ic

# file to save the advected fields
adv_fields_fn = Path(save_dir) / f"adv_fields_nmesh{nmesh:d}.asdf"
adv_power_fn = Path(save_dir) / f"adv_power_nmesh{nmesh:d}.asdf"
print(str(adv_fields_fn))
print(str(adv_power_fn))


# load fields

# fields_fn = Path(save_dir) / f"fields_nmesh{nmesh:d}.asdf"
# f = asdf.open(fields_fn, lazy_load=False)

path = '/pscratch/sd/r/rhliu/projects/heft_scratch/MillenniumTNG_sims/'

dens = np.load(path + 'density_ngenic.npy')
# tau = np.load(path + 'tau_3d_snap_264.npy')
tau = load_tau(z_mock, path)

# Create offset for the density mesh (Important)
print('Create offset for the density mesh')
n4 = nmesh//4
density_ngen = dens.copy()
for k in range(nmesh):
    # print(k)
    density_2d_ngen = density_ngen[:, :, k]

    # shifts up by L/4 in y
    tmp1 = density_2d_ngen[:(nmesh - n4), :]
    tmp2 = density_2d_ngen[(nmesh - n4):, :]
    density_2d_ngen[n4:, :] = tmp1
    density_2d_ngen[:n4, :] = tmp2
    density_ngen[:, :, k] = density_2d_ngen
dens = density_ngen.copy()

d, d2, s2, n2 = get_fields(dens, Lbox, nmesh)
table = {}
table['delta'] = d
table['delta2'] = d2
table['nabla2'] = n2
table['tidal2'] = s2

# Now we calculate the advected fields:
adv_fields = {}
for field in factors_fields.keys():
    adv_fields[field] = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
adv_fields['1cb'] = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)


# load fields
print('Load Fields', time.time()-t0)

# path = '/pscratch/sd/b/boryanah/for_henry/'
# path = '/pscratch/sd/r/rhliu/projects/heft_scratch/MillenniumTNG_sims/'

lagr_pos = load_lagrangians(path=path)
pcle_pos = load_positions(z=z_mock, path=path)
velocity = load_velocities(z=z_mock, path=path)

# VERY IMPORTANT LINE: position offsets
# Either do this for these fields or for the IC field (dens) above, better to do it above for dens I believe.

# lagr_pos[:, 0] -= Lbox/4
# lagr_pos = lagr_pos % Lbox
# pcle_pos[:, 0] -= Lbox/4
# pcle_pos = pcle_pos % Lbox

print('Calculate Advected Fields', time.time()-t0)

# lagr_ijk = ((lagr_pos+Lbox/2.)/(Lbox/nmesh)).astype(int)%nmesh # better than without L/2
lagr_ijk = ((lagr_pos)/(Lbox/nmesh)).astype(int)%nmesh # better than without L/2
for field in factors_fields.keys():
    # w = (f['data'][field]*D_ratio**factors_fields[field])[lagr_ijk[:,0], lagr_ijk[:,1], lagr_ijk[:,2]]
    w = (table[field]*D_ratio**factors_fields[field])[lagr_ijk[:,0], lagr_ijk[:,1], lagr_ijk[:,2]]

    if paste == "TSC":
        tsc_parallel(pcle_pos, adv_fields[field], Lbox, weights=w)

if paste == "TSC":
    tsc_parallel(pcle_pos, adv_fields['1cb'], Lbox, weights=None)


header = {}
header['sim_name'] = sim_name
header['Lbox'] = Lbox
header['pcle_type'] = pcle_type # could be A or B
header['paste'] = paste
header['nmesh'] = nmesh
header['kcut'] = kcut
# compress_asdf(str(adv_fields_fn), adv_fields, header)



print('now for fitting tau', time.time()-t0)

# Parameters
# k_bin_edges = np.linspace(0.0, 1., 21) # best
k_bin_edges = np.linspace(1e-2, 10., 21) # best
k_binc = (k_bin_edges[1:]+k_bin_edges[:-1])*.5
mu_bin_edges = np.array([0, 1.])
mu_binc = (mu_bin_edges[1:]+mu_bin_edges[:-1])*.5
sim_name = "MillenniumTNG"
z_IC = z_ic


# load advected
# our advected fields - HL

ones_dm_advected = adv_fields['1cb']
delta_dm_advected = adv_fields['delta']
delta_dm_squared_advected = adv_fields['delta2']
s2_dm_advected = adv_fields['tidal2']
nabla2_dm_advected = adv_fields['nabla2']

delta_tau = tau/np.mean(tau) - 1 # Question: Is this advected?

# normalize Question: Is everything normalized from before? I assume so
# ones_dm_advected /= np.mean(ones_dm_advected, dtype=np.float64)
# ones_dm_advected -= 1.
# delta_dm_advected /= (npart/nmesh)**3
# delta_dm_squared_advected /= (npart/nmesh)**3
# s2_dm_advected /= (npart/nmesh)**3
# nabla2_dm_advected /= (npart/nmesh)**3

normalize_mean = np.mean(ones_dm_advected, dtype=np.float64)
ones_dm_advected /= normalize_mean
ones_dm_advected -= 1.
delta_dm_advected /= normalize_mean
delta_dm_squared_advected /= normalize_mean
s2_dm_advected /= normalize_mean
nabla2_dm_advected /= normalize_mean

# fourier transform
delta_tau_obs_fft = rfftn(delta_tau, workers=-1) / np.complex64(delta_tau.size)

ones_dm_advected_fft = rfftn(ones_dm_advected, workers=-1) / np.complex64(ones_dm_advected.size)
delta_dm_advected_fft = rfftn(delta_dm_advected, workers=-1) / np.complex64(delta_dm_advected.size)
delta_dm_squared_advected_fft = rfftn(delta_dm_squared_advected, workers=-1) / np.complex64(delta_dm_squared_advected.size)
s2_dm_advected_fft = rfftn(s2_dm_advected, workers=-1) / np.complex64(s2_dm_advected.size)
nabla2_dm_advected_fft = rfftn(nabla2_dm_advected, workers=-1) / np.complex64(nabla2_dm_advected.size)

# del delta_tau, ones_dm_advected, delta_dm_advected, delta_dm_squared_advected, s2_dm_advected, nabla2_dm_advected
# gc.collect()

# get auto-power spectrum
result = calc_pk_from_deltak(delta_tau_obs_fft, Lbox, k_bin_edges, mu_bin_edges)

pk = result['power']
Nmode = result['N_mode']
binned_poles = result['binned_poles']
N_mode_poles = result['N_mode_poles']
k_avg = result['k_avg']
#np.savez(f"data/power_21_obs_fft_snap{snap_int:d}.npz", pk=pk, k_bin_edges=k_bin_edges, Nmode=Nmode, k_avg=k_avg)

# just because I'm lazy and want to treat both options the same way
if len(mu_binc) == 1:
    pk_tau = np.atleast_2d(pk).T
    k_avg = np.atleast_2d(k_avg).T
    Nmode = np.atleast_2d(Nmode).T
else:
    pk_tau = pk

# get cross-power spectra
power_dict = {}
fields = ["ones_dm_advected", "delta_dm_advected", "delta_dm_squared_advected", "s2_dm_advected", "nabla2_dm_advected"]
for i, field_i in enumerate(fields):
    result = calc_pk_from_deltak(delta_tau_obs_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=locals()[f"{field_i}_fft"])

    pk = result['power']
    Nmode = result['N_mode']
    binned_poles = result['binned_poles']
    N_mode_poles = result['N_mode_poles']
    k_avg = result['k_avg']   
    if len(mu_binc) == 1:
        pk = np.atleast_2d(pk).T
        k_avg = np.atleast_2d(k_avg).T
        Nmode = np.atleast_2d(Nmode).T
    power_dict[f"delta_tau_obs_{field_i}"] = pk
    
    for j, field_j in enumerate(fields):
        if i < j: continue
        result = calc_pk_from_deltak(locals()[f"{field_i}_fft"], Lbox, k_bin_edges, mu_bin_edges, field2_fft=locals()[f"{field_j}_fft"])

        pk = result['power']
        Nmode = result['N_mode']
        binned_poles = result['binned_poles']
        N_mode_poles = result['N_mode_poles']
        k_avg = result['k_avg']
        if len(mu_binc) == 1:
            pk = np.atleast_2d(pk).T
            k_avg = np.atleast_2d(k_avg).T
            Nmode = np.atleast_2d(Nmode).T
        power_dict[f"{field_i}_{field_j}"] = power_dict[f"{field_j}_{field_i}"] = pk
#np.savez(f"data/power_dict_snap{snap_int:d}.npz", power_dict=power_dict, k_bin_edges=k_bin_edges, mu_bin_edges=mu_bin_edges, Nmode=Nmode, k_avg=k_avg)


Fit_fn = HEFTFit(ones_dm_advected, delta_dm_advected, delta_dm_squared_advected, 
                 s2_dm_advected, nabla2_dm_advected, delta_tau, Lbox=500, nmesh=1080)

dict_list = []
for options in ['field-level-brute', 'field-level-scale', 'field-level-matrix', 'power-spectrum']:
    
    # print(options)
    dict_i = Fit_fn.fit(options, kmax=1.0, save=False, return_val=True)
    dict_list.append(dict_i)

print(dict_list[0].keys())

r_pk = dict_list[0]['r_pk']
r_pk_bk = dict_list[1]['r_pk']
r_pk_alt = dict_list[2]['r_pk']
r_pk_fit = dict_list[3]['r_pk']

pk_mod = dict_list[0]['pk_mod']
pk_mod_bk = dict_list[1]['pk_mod']
pk_mod_alt = dict_list[2]['pk_mod']
pk_mod_fit = dict_list[3]['pk_mod']
pk_tau = Fit_fn.pk_tau

pk_err = dict_list[0]['pk_err']
pk_err_bk = dict_list[1]['pk_err']
pk_err_alt = dict_list[2]['pk_err']
pk_err_fit = dict_list[3]['pk_err']

# relevant plots are everything except thermal - HL

plt.figure(1, figsize=(9, 7))
plt.figure(2, figsize=(9, 7))
plt.figure(3, figsize=(9, 7))
for i in range(1):

    plt.figure(1)
    plt.title(f"r_cc, z = {z_mock:.1f}")
    plt.plot(k_avg[:, i], r_pk[:, i], label="field-level-brute")
    plt.plot(k_avg[:, i], r_pk_bk[:, i], label="field-level-scale")
    plt.plot(k_avg[:, i], r_pk_alt[:, i], label="field-level-matrix")
    # plt.plot(k_avg[:, i], r_pk_fit[:, i], label="power-spectrum")
    plt.legend()
    plt.xscale('log')
    plt.xlabel('k')
    plt.ylabel('r(k)')
    # plt.ylim([0, 1.0]) 

    plt.figure(2)
    plt.title(f"Power Spectrum, z = {z_mock:.1f}")
    plt.plot(k_avg[:, i], pk_mod[:, i]*k_avg[:, i]**3/2./np.pi**2, label="field-level-brute")
    plt.plot(k_avg[:, i], pk_mod_bk[:, i]*k_avg[:, i]**3/2./np.pi**2, label="field-level-scale")
    plt.plot(k_avg[:, i], pk_mod_alt[:, i]*k_avg[:, i]**3/2./np.pi**2, label="field-level-matrix")
    plt.plot(k_avg[:, i], pk_mod_fit[:, i]*k_avg[:, i]**3/2./np.pi**2, label="power-spectrum")
    plt.errorbar(k_avg[:, i], pk_tau[:, i]*k_avg[:, i]**3/2./np.pi**2, yerr=np.sqrt(2./Nmode[:, i])*pk_tau[:, i]*k_avg[:, i]**3/2./np.pi**2, capsize=4, label="Tau")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('P(k)')

    plt.figure(3)
    plt.title(f"Pk error ratio, z = {z_mock:.1f}")
    plt.plot(k_avg[:, i], np.abs(pk_err/pk_tau)[:, i], label="field-level-brute")
    plt.plot(k_avg[:, i], np.abs(pk_err_bk/pk_tau)[:, i], label="field-level-scale")
    plt.plot(k_avg[:, i], np.abs(pk_err_alt/pk_tau)[:, i], label="field-level-matrix")
    plt.plot(k_avg[:, i], np.abs(pk_err_fit/pk_tau)[:, i], label="power-spectrum")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    # plt.ylim([0, 0.5]) 
    plt.ylabel('P_err(k)/P_tau(k)')

plt.figure(1)
plt.savefig("../figures/cross_corr_z0.png")
plt.close()
plt.figure(2)
plt.savefig("../figures/power_tau_z0.png")
plt.close()
plt.figure(3)
plt.savefig("../figures/error_ratio_z0.png")
plt.close()