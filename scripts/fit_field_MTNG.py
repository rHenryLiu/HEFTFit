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
z_ic = 49
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
tau = np.load(path + 'tau_3d_snap_264.npy')

# Create offset for the density mesh
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

def load_lagrangians():
    lagrangians = []
    for i in range(10):
        i = i+1
        # lagrangians.append(np.load(path + 'lagrangian_position_sorted_264_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
        lagrangians.append(np.load(path + 'position_sorted_000_MTNG-L500-1080-A_part{}_of_10.npy'.format(i)))
    return np.concatenate(lagrangians)

def load_positions(z=z_mock):
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

def load_velocities(z=z_mock):
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

lagr_pos = load_lagrangians()
pcle_pos = load_positions(z=z_mock)
velocity = load_velocities(z=z_mock)

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


print('Save Advected Fields', time.time()-t0)

header = {}
header['sim_name'] = sim_name
header['Lbox'] = Lbox
header['pcle_type'] = pcle_type # could be A or B
header['paste'] = paste
header['nmesh'] = nmesh
header['kcut'] = kcut
# compress_asdf(str(adv_fields_fn), adv_fields, header)



############################################################################################
# # get_power_MTNG.py file
# print('now for get_power_MTNG.py', time.time()-t0)

# # define k bins
# # k_bin_edges = np.linspace(0, 1., 201) # h/Mpc
# k_bin_edges = np.linspace(1e-2, 1., 201) # h/Mpc
# mu_bin_edges = np.array([0., 1.]) # angle

# k_binc = (k_bin_edges[1:] + k_bin_edges[:-1])*.5
# mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1])*.5

# fields = ['1cb', 'delta', 'delta2', 'nabla2', 'tidal2']
# # loop over fields # todo!!!! better to save field_fft
# adv_pk_dict = {}
# for i, field in enumerate(fields):
#     # get the fft field
#     # tuks
#     if field == '1cb':
#         field_fft = rfftn(adv_fields[field]/np.mean(adv_fields[field], dtype=np.float64) - np.float32(1.), workers=-1)/np.complex64(adv_fields[field].size)
#     else:
#         field_fft = rfftn(adv_fields[field]/np.mean(adv_fields['1cb'], dtype=np.float64), workers=-1)/np.complex64(adv_fields[field].size)

#     for j, field2 in enumerate(fields):
#         if i < j: continue

#         # get the fft field2
#         if field2 == "1cb":
#             field2_fft = rfftn(adv_fields[field2]/np.mean(adv_fields[field2], dtype=np.float64) - np.float32(1.), workers=-1)/np.complex64(adv_fields[field2].size)
#         else:
#             field2_fft = rfftn(adv_fields[field2]/np.mean(adv_fields['1cb'], dtype=np.float64), workers=-1)/np.complex64(adv_fields[field2].size)

#         # compute power spectrum
#         adv_pk_dict[f'{field}_{field2}'] = calc_pk_from_deltak(field_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft)
#         adv_pk_dict[f'{field}_{field2}']['k_min'] = k_bin_edges[:-1]
#         adv_pk_dict[f'{field}_{field2}']['k_max'] = k_bin_edges[1:]
#         adv_pk_dict[f'{field}_{field2}']['k_mid'] = k_binc
#         adv_pk_dict[f'{field2}_{field}'] = adv_pk_dict[f'{field}_{field2}']

#         # del field2_fft
#         # gc.collect()

# # del field_fft
# # gc.collect()
# print(adv_pk_dict.keys())
# # save fields using asdf compression
# header = {}
# header['sim_name'] = sim_name
# header['Lbox'] = Lbox
# header['pcle_type'] = pcle_type # could be A or B
# header['paste'] = paste
# header['nmesh'] = nmesh
# header['kcut'] = kcut
# # compress_asdf(str(adv_power_fn), adv_pk_dict, header)

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
print('now for fitting tau', time.time()-t0)

# Parameters
# k_bin_edges = np.linspace(0.0, 1., 21) # best
k_bin_edges = np.linspace(1e-2, 1., 21) # best
k_binc = (k_bin_edges[1:]+k_bin_edges[:-1])*.5
mu_bin_edges = np.array([0, 1.])
mu_binc = (mu_bin_edges[1:]+mu_bin_edges[:-1])*.5
sim_name = "MillenniumTNG"
z_IC = z_ic

sim_str = f"_{sim_name}"
snap_int = 40 # 23, 34, 40, 48, 54, 61, 70, 76
snap_dict = {0: [z_IC, -1], 40: [8.3, 168], 48: [7.5, 209], 54: [7.0, 240], 75: [5.75, 342], 70: [6.0, 318], 61: [6.5, 276], 34: [9.0, 139], 23: [10.82, 83] }
scratch_dir = heft_dir
sim_str = f"_{sim_name}"
snap_int = 40 # 23, 34, 40, 48, 54, 61, 70, 76
snap_dict = {0: [z_IC, -1], 40: [8.3, 168], 48: [7.5, 209], 54: [7.0, 240], 75: [5.75, 342], 70: [6.0, 318], 61: [6.5, 276], 34: [9.0, 139], 23: [10.82, 83] }

# nmesh = 1024
# npart = 2100 # Question: Whats this?
# Lbox = 64.6917  # cMpc/h
# k_bin_edges = np.linspace(0.0, 1., 21) # best
# #k_bin_edges = np.linspace(0.0, 1., 11) # rly bad
# #k_bin_edges = (np.arange(20) + 0.5) * 2 * np.pi / (Lbox) # qin
# k_binc = (k_bin_edges[1:]+k_bin_edges[:-1])*.5
# mu_bin_edges = np.array([0, 1.])
# #mu_bin_edges = np.linspace(0, 1., 21)
# mu_binc = (mu_bin_edges[1:]+mu_bin_edges[:-1])*.5
# #sim_name = "Thesan-Dark-3" # 525
# #sim_name = "Thesan-Dark-2" # 1050
# sim_name = "Thesan-Dark-1" # 1024, only down so far
# #sim_name = "Thesan-Dark-1_z20" # noisy
# #sim_name = "Thesan-Dark-1_z49" # noisy
# if "z20" in sim_name:    
#     z_IC = 20.
# else:
#     z_IC = 49.
# sim_str = f"_{sim_name}"
# snap_int = 40 # 23, 34, 40, 48, 54, 61, 70, 76
# snap_dict = {0: [z_IC, -1], 40: [8.3, 168], 48: [7.5, 209], 54: [7.0, 240], 75: [5.75, 342], 70: [6.0, 318], 61: [6.5, 276], 34: [9.0, 139], 23: [10.82, 83] }
# scratch_dir = "/pscratch/sd/b/boryanah/Thesan/derived/"
# thesan_dir = "/pscratch/sd/b/boryanah/Thesan/"
# want_rsd = True #False
# rsd_str = "_rsd" if want_rsd else ""
# norsd_str = "_norsd" if not want_rsd else ""
# want_down = True
# if want_down:
#     assert sim_name == "Thesan-Dark-1"
#     down_str = "_down"
# else:
#     down_str = ""


# load advected
# our advected fields - HL

ones_dm_advected = adv_fields['1cb']
delta_dm_advected = adv_fields['delta']
delta_dm_squared_advected = adv_fields['delta2']
s2_dm_advected = adv_fields['tidal2']
nabla2_dm_advected = adv_fields['nabla2']

delta_tau_advected = tau/np.mean(tau) - 1 # Question: Is this advected?


# ones_dm_advected = np.load(scratch_dir+f'ones_dm{rsd_str}_advected_{snap_int:d}{sim_str}{down_str}.npy')
# delta_dm_advected = np.load(scratch_dir+f'delta_dm{rsd_str}_advected_{snap_int:d}{sim_str}{down_str}.npy')
# delta_dm_squared_advected = np.load(scratch_dir+f'delta_dm_squared{rsd_str}_advected_{snap_int:d}{sim_str}{down_str}.npy')
# s2_dm_advected = np.load(scratch_dir+f's2_dm{rsd_str}_advected_{snap_int:d}{sim_str}{down_str}.npy')
# nabla2_dm_advected = np.load(scratch_dir+f'nabla2_dm{rsd_str}_advected_{snap_int:d}{sim_str}{down_str}.npy')



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
delta_tau_obs_fft = rfftn(delta_tau_advected, workers=-1) / np.complex64(delta_tau_advected.size)

ones_dm_advected_fft = rfftn(ones_dm_advected, workers=-1) / np.complex64(ones_dm_advected.size)
delta_dm_advected_fft = rfftn(delta_dm_advected, workers=-1) / np.complex64(delta_dm_advected.size)
delta_dm_squared_advected_fft = rfftn(delta_dm_squared_advected, workers=-1) / np.complex64(delta_dm_squared_advected.size)
s2_dm_advected_fft = rfftn(s2_dm_advected, workers=-1) / np.complex64(s2_dm_advected.size)
nabla2_dm_advected_fft = rfftn(nabla2_dm_advected, workers=-1) / np.complex64(nabla2_dm_advected.size)

# del delta_tau_advected, ones_dm_advected, delta_dm_advected, delta_dm_squared_advected, s2_dm_advected, nabla2_dm_advected
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

# define model for auto and cross
# one_bias is the list of biases - HL (1, b1, b2, bs, bnabla)
def get_power_model(one_bias, kmin=0., kmax=np.inf, mumax=np.inf):
    power_model = np.zeros((np.sum((k_binc <= kmax) & (k_binc > kmin)), np.sum(mu_binc < mumax)))
    for i, field_i in enumerate(fields):
        for j, field_j in enumerate(fields):
            power_model += one_bias[i] * one_bias[j] * power_dict[f"{field_i}_{field_j}"][((k_binc <= kmax) & (k_binc > kmin))[:, None] & (mu_binc < mumax)[None, :]].reshape(np.sum(((k_binc <= kmax) & (k_binc > kmin))), np.sum(mu_binc < mumax))
    return power_model

def get_cross_power_model(one_bias, kmin=0., kmax=np.inf, mumax=np.inf):
    power_model = np.zeros((np.sum((k_binc <= kmax) & (k_binc > kmin)), np.sum(mu_binc < mumax)))
    for i, field_i in enumerate(fields):
        power_model += one_bias[i] * power_dict[f"delta_tau_obs_{field_i}"][((k_binc <= kmax) & (k_binc > kmin))[:, None] & (mu_binc < mumax)[None, :]].reshape(np.sum(((k_binc <= kmax) & (k_binc > kmin))), np.sum(mu_binc < mumax))
    return power_model

# needed in field-level fits
kx, ky, kz = fftfreq(nmesh, d=Lbox/nmesh)*2.*np.pi, fftfreq(nmesh, d=Lbox/nmesh)*2.*np.pi, rfftfreq(nmesh, d=Lbox/nmesh)*2.*np.pi
kx, ky, kz = np.meshgrid(kx, ky, kz)#, indexing='ij') # TESTING
k2 = kx**2 + ky**2 + kz**2
mu2 = kz**2/k2
mu2[0, 0, 0] = 0.
del kx, ky, kz; gc.collect()

options = 'field-level-brute' # Brute force, above mini_fun
options = 'field-level-scale' # Field level scale dependence
options = 'field-level-matrix' # From 2112.00012
options = 'power-spectrum' # straightforward power spectra

Fit_fn = HEFTFit(ones_dm_advected, delta_dm_advected, delta_dm_squared_advected, 
                 s2_dm_advected, nabla2_dm_advected, delta_tau_advected, Lbox=500, nmesh=1080)

dict_list = []
for options in ['field-level-brute', 'field-level-scale', 'field-level-matrix', 'power-spectrum']:
    
    print(options)
    dict_i = Fit_fn.fit(options, save=False, return_val=True)
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

# Testing at power-spectrum level


# loading noise vurves
# dont worry about this - HL

# k, mu, pk_7, pk_7p5, pk_8, pk_8p5, pk_9 = np.loadtxt("/global/homes/b/boryanah/Thesan/data/p21_thermal_1000detectors_5years.txt", unpack=True)
# mu_uni = np.unique(mu)

# plt.figure(1, figsize=(9, 7))
# for i in range(len(mu_binc)):
#     #if i != 19: continue
#     i_argmin = np.argmin(np.abs(mu_uni-mu_binc[i]))
#     pk_thermal = np.interp(k_avg[:, i], k[np.isclose(mu, mu_uni[i_argmin])], pk_8p5[np.isclose(mu, mu_uni[i_argmin])])
#     mask = ~np.isclose(pk_tau[:, i], 0.)
    
#     ratio = np.zeros(len(k_avg[:, i]))
#     ratio[mask] = (pk_err[mask, i]/pk_tau[mask, i])
#     plt.plot(k_avg[:, i], ratio, lw=2, color='darkblue')

#     ratio = np.zeros(len(k_avg[:, i]))
#     ratio[mask] = (pk_err_bk[mask, i]/pk_tau[mask, i])
#     plt.plot(k_avg[:, i], ratio, lw=2, color='darkgreen')

#     ratio = np.zeros(len(k_avg[:, i]))
#     ratio[mask] = pk_thermal[mask]/pk_tau[mask, i]
#     plt.plot(k_avg[:, i], ratio, color='red')

# plt.gca().axhline(y=0.1, ls='--', lw=2, color='k')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$k \ [h/{\rm Mpc}]$', fontsize=20)
# plt.ylabel(r'$P_{\rm err}(k)/P_{21}(k)$', fontsize=20)
# plt.ylim([0.01, 0.5]) 
# #plt.ylim([0.005, 0.5]) 
# plt.xlim([0.05, 1.01])
# plt.savefig("../figures/thermal_comparison.png")
# plt.close()

# plt.figure(3, figsize=(9, 7))
# i = 0
# plt.plot(k_avg[:, i], np.abs(pk_err/pk_tau)[:, i], lw=2, color='darkblue')
# plt.plot(k_avg[:, i], np.abs(pk_err_bk/pk_tau)[:, i], lw=2, color='darkgreen')
# #plt.legend()
# plt.gca().axhline(y=0.1, ls='--', lw=2, color='k')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$k \ [h/{\rm Mpc}]$', fontsize=20)
# plt.ylabel(r'$P_{\rm err}(k)/P_{21}(k)$', fontsize=20)
# #plt.ylim([0.01, 0.5]) 
# plt.ylim([0.005, 0.5]) 
# plt.xlim([0.05, 1.01])
# plt.savefig("../figures/thermal_comparison.png")
# plt.close()

# relevant plots are everything except thermal - HL

plt.figure(1, figsize=(9, 7))
plt.figure(2, figsize=(9, 7))
plt.figure(3, figsize=(9, 7))
# print(k_avg[:, 0])
# print(pk_tau[:, 0])
# print(pk_mod[:, 0])
# print(pk_mod.shape)
for i in range(1):

    plt.figure(1)
    # plt.title(f"z = {snap_dict[snap_int][0]:.1f}")
    plt.plot(k_avg[:, i], r_pk[:, i], label="field-level-brute")
    plt.plot(k_avg[:, i], r_pk_bk[:, i], label="field-level-scale")
    plt.plot(k_avg[:, i], r_pk_alt[:, i], label="field-level-matrix")
    plt.plot(k_avg[:, i], r_pk_fit[:, i], label="power-spectrum")
    plt.legend()
    plt.xscale('log')
    plt.xlabel('k')
    plt.ylabel('r(k)')
    plt.ylim([0, 1.0]) 

    plt.figure(2)
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
    # plt.plot(k_avg[:, i], np.abs((pk_mod-pk_tau)/pk_tau)[:, i], label="field-level")
    # plt.plot(k_avg[:, i], np.abs((pk_mod_fit-pk_tau)/pk_tau)[:, i], label="power-spectrum")
    plt.plot(k_avg[:, i], np.abs(pk_err/pk_tau)[:, i], label="field-level-brute")
    plt.plot(k_avg[:, i], np.abs(pk_err_bk/pk_tau)[:, i], label="field-level-scale")
    plt.plot(k_avg[:, i], np.abs(pk_err_alt/pk_tau)[:, i], label="field-level-matrix")
    #plt.plot(k_binc, np.abs(pk_err/pk_tau)[:, i], label="field-level") # why are these different
    plt.plot(k_avg[:, i], np.abs(pk_err_fit/pk_tau)[:, i], label="power-spectrum")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    # plt.ylim([0, 0.5]) 
    plt.ylabel('P_err(k)/P_tau(k)')

plt.figure(1)
plt.savefig("../figures/cross_corr6.png")
plt.close()
plt.figure(2)
plt.savefig("../figures/power_tau6.png")
plt.close()
plt.figure(3)
plt.savefig("../figures/error_ratio6.png")
plt.close()

print(r_pk_fit[:, i])