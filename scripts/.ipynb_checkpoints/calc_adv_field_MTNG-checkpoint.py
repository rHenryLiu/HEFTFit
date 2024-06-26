import argparse
import gc
import os
from pathlib import Path
import warnings

import asdf
import numpy as np
import matplotlib.pyplot as plt
import yaml
import numba
from scipy.fft import rfftn, irfftn

from classy import Class
# from abacusnbody.metadata import get_meta
from abacusnbody.analysis.tsc import tsc_parallel #put it on a grid using tsc interpolation
from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak #computes power spectrum from density contrast, not specific to abacus 
import time

from asdf.exceptions import AsdfWarning
warnings.filterwarnings('ignore', category=AsdfWarning)

import sys
sys.path.append('../velocileptors') # clone the velocileptors from github 
from velocileptors.LPT.cleft_fftw import CLEFT
sys.path.append('../')
from HEFTFit.field_utils import *

# DEFAULTS = {'path2config': 'config/abacus_heft.yaml'}

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
# z_ic = 49.
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

ic_path = '/pscratch/sd/r/rhliu/projects/heft_scratch/MillenniumTNG_sims/density_ngenic.npy'

dens = np.load(ic_path)

# Create offset for the density mesh
# n4 = nmesh//4
# density_ngen = dens.copy()
# for k in range(nmesh):
#     print(k)
#     density_2d_ngen = density_ngen[:, :, k]

#     # shifts up by L/4 in y
#     tmp1 = density_2d_ngen[:(nmesh - n4), :]
#     tmp2 = density_2d_ngen[(nmesh - n4):, :]
#     density_2d_ngen[n4:, :] = tmp1
#     density_2d_ngen[:n4, :] = tmp2
#     density_ngen[:, :, k] = density_2d_ngen
# dens = density_ngen.copy()

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
path = '/pscratch/sd/r/rhliu/projects/heft_scratch/MillenniumTNG_sims/'

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
pcle_pos = load_positions()
velocity = load_velocities()

# VERY IMPORTANT LINE: position offsets

# lagr_pos[:, 0] -= Lbox/4
lagr_pos[:, 0] -= Lbox/4
lagr_pos = lagr_pos % Lbox
pcle_pos[:, 0] -= Lbox/4
pcle_pos = pcle_pos % Lbox

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
# get_power_MTNG.py file
print('now for get_power_MTNG.py', time.time()-t0)

# define k bins
# k_bin_edges = np.linspace(0, 1., 201) # h/Mpc
k_bin_edges = np.linspace(1e-2, 1., 201) # h/Mpc
mu_bin_edges = np.array([0., 1.]) # angle

k_binc = (k_bin_edges[1:] + k_bin_edges[:-1])*.5
mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1])*.5

fields = ['1cb', 'delta', 'delta2', 'nabla2', 'tidal2']
# loop over fields # todo!!!! better to save field_fft
adv_pk_dict = {}
for i, field in enumerate(fields):
    # get the fft field
    # tuks
    if field == '1cb':
        field_fft = rfftn(adv_fields[field]/np.mean(adv_fields[field], dtype=np.float64) - np.float32(1.), workers=-1)/np.complex64(adv_fields[field].size)
    else:
        field_fft = rfftn(adv_fields[field]/np.mean(adv_fields['1cb'], dtype=np.float64), workers=-1)/np.complex64(adv_fields[field].size)

    for j, field2 in enumerate(fields):
        if i < j: continue

        # get the fft field2
        if field2 == "1cb":
            field2_fft = rfftn(adv_fields[field2]/np.mean(adv_fields[field2], dtype=np.float64) - np.float32(1.), workers=-1)/np.complex64(adv_fields[field2].size)
        else:
            field2_fft = rfftn(adv_fields[field2]/np.mean(adv_fields['1cb'], dtype=np.float64), workers=-1)/np.complex64(adv_fields[field2].size)

        # compute power spectrum
        adv_pk_dict[f'{field}_{field2}'] = calc_pk_from_deltak(field_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft)
        adv_pk_dict[f'{field}_{field2}']['k_min'] = k_bin_edges[:-1]
        adv_pk_dict[f'{field}_{field2}']['k_max'] = k_bin_edges[1:]
        adv_pk_dict[f'{field}_{field2}']['k_mid'] = k_binc
        adv_pk_dict[f'{field2}_{field}'] = adv_pk_dict[f'{field}_{field2}']

        # del field2_fft
        # gc.collect()

# del field_fft
# gc.collect()
print(adv_pk_dict.keys())
# save fields using asdf compression
header = {}
header['sim_name'] = sim_name
header['Lbox'] = Lbox
header['pcle_type'] = pcle_type # could be A or B
header['paste'] = paste
header['nmesh'] = nmesh
header['kcut'] = kcut
# compress_asdf(str(adv_power_fn), adv_pk_dict, header)


############################################################################################
# compare_theory.py file
print('now for compare_theory.py', time.time()-t0)

# load input linear power
# kth = meta['CLASS_power_spectrum']['k (h/Mpc)']
# pk_z1 = meta['CLASS_power_spectrum']['P (Mpc/h)^3']

# Compare to Velocileptors
k_max = 10.
kth = np.logspace(-2, np.log10(k_max), num=1000) #Mpc^-1
z = z_mock
# pk_z1 = np.array([pkclass.pk_lin(ki, z) for ki in kth])


# rewind back to interest redshift of the simulation
# pk_cb = (D_mock/D_z1)**2*pk_z1
pk_cb = np.array([pkclass.pk_lin(ki, z) for ki in kth])

# Unit conversion (Important!)
kth /= h
pk_cb *= h**3

# apply gaussian cutoff to linear power
if not np.isclose(kcut, 0.):
    pk_cb *= np.exp(-(kth/kcut)**2)

# Initialize the class -- with no wisdom file passed it will
# experiment to find the fastest FFT algorithm for the system.
# B.H. modified velocileptors/Utils/loginterp.py in case of exception error
cleft = CLEFT(kth, pk_cb, cutoff=10)
# You could save the wisdom file here if you wanted:
# mome.export_wisdom(wisdom_file_name)

# The first four are deterministic Lagrangian bias up to third order
# While alpha and sn are the counterterm and stochastic term (shot noise)
cleft.make_ptable()
kv = cleft.pktable[:, 0]

# parse velocileptors
pk_theo = {'1cb_1cb': cleft.pktable[:, 1],\
           '1cb_delta': 0.5*cleft.pktable[:, 2], 'delta_delta': cleft.pktable[:, 3],\
           '1cb_delta2': 0.5*cleft.pktable[:, 4], 'delta_delta2': 0.5*cleft.pktable[:, 5],  'delta2_delta2': cleft.pktable[:, 6],\
           '1cb_tidal2': 0.5*cleft.pktable[:, 7], 'delta_tidal2': 0.5*cleft.pktable[:, 8],  'delta2_tidal2': 0.5*cleft.pktable[:, 9],\
           'tidal2_tidal2': cleft.pktable[:, 10], '1cb_nabla2': kv**2*0.5*cleft.pktable[:, 2],\
           'delta_nabla2': kv**2*cleft.pktable[:, 3], 'nabla2_nabla2': np.interp(kv, kth, pk_cb*kth**2),
           'nabla2_tidal2': kv**2*0.5*cleft.pktable[:, 8], 'delta2_nabla2': kv**2*0.5*cleft.pktable[:, 5]}
"""
           'delta_nabla2': np.interp(kv, kth, pk_cb*kth**2), 'nabla2_nabla2': np.interp(kv, kth, pk_cb*kth**2),
           'nabla2_tidal2': np.interp(kv, kth, pk_cb*kth**2), 'delta2_nabla2': np.interp(kv, kth, pk_cb*kth**2)}
"""
# just so we don't have to worry
pk_theo_copy = {} # lazy
for key in pk_theo.keys():
    pk_theo_copy[key] = pk_theo[key]
for pk in pk_theo_copy.keys():
    pk_theo[f"{pk.split('_')[1]}_{pk.split('_')[0]}"] = pk_theo[pk]

# load power
# pk_data = asdf.open(adv_power_fn, lazy_load=False)['data']
pk_data = adv_pk_dict


# Plotting
hexcols = np.array(['#44AA99', '#117733', '#999933', '#88CCEE', '#332288', '#BBBBBB', '#4477AA',
                    '#CC6677', '#AA4499', '#6699CC', '#AA4466', '#882255', '#661100', '#0099BB', '#DDCC77'])


plt.subplots(2, 3, figsize=(16,9))
count = 0
for i, field in enumerate(fields):
    for j, field2 in enumerate(fields):
        if i < j: continue
        #plt.subplot(2, 3, count % 6 + 1)
        # TODO make pretty just to make comparison easier
        if "1cb_1cb" == f"{field}_{field2}": plt.subplot(2, 3, 1)
        if ("1cb_delta" == f"{field}_{field2}") or ("1cb_delta" == f"{field2}_{field}"): plt.subplot(2, 3, 1)
        if "delta_delta" == f"{field}_{field2}": plt.subplot(2, 3, 2)
        if ("delta_delta2" == f"{field}_{field2}") or ("delta_delta2" == f"{field2}_{field}"): plt.subplot(2, 3, 2)
        if "delta2_delta2" == f"{field}_{field2}": plt.subplot(2, 3, 3)
        if ("delta2_tidal2" == f"{field}_{field2}") or ("delta2_tidal2" == f"{field2}_{field}"): plt.subplot(2, 3, 3)
        if ("delta2_nabla2" == f"{field}_{field2}") or ("delta2_nabla2" == f"{field2}_{field}"): plt.subplot(2, 3, 3)
        if ("1cb_delta2" == f"{field}_{field2}") or ("1cb_delta2" == f"{field2}_{field}"): plt.subplot(2, 3, 4)
        if ("1cb_tidal2" == f"{field}_{field2}") or ("1cb_tidal2" == f"{field2}_{field}"): plt.subplot(2, 3, 4)
        if ("1cb_nabla2" == f"{field}_{field2}") or ("1cb_nabla2" == f"{field2}_{field}"): plt.subplot(2, 3, 4)
        if ("delta_tidal2" == f"{field}_{field2}") or ("delta_tidal2" == f"{field2}_{field}"): plt.subplot(2, 3, 5)
        if ("delta_nabla2" == f"{field}_{field2}") or ("delta_nabla2" == f"{field2}_{field}"): plt.subplot(2, 3, 5)
        if "nabla2_nabla2" == f"{field}_{field2}": plt.subplot(2, 3, 6)
        if "tidal2_tidal2" == f"{field}_{field2}": plt.subplot(2, 3, 6)
        if ("tidal2_nabla2" == f"{field}_{field2}") or ("tidal2_nabla2" == f"{field2}_{field}"): plt.subplot(2, 3, 6)

        k_mid = pk_data[f"{field}_{field2}"]['k_mid'].flatten()
        Pk_data = pk_data[f"{field}_{field2}"]['power'].flatten()
        Pk_theo = pk_theo[f"{field}_{field2}"]

        # LPT defines 1/2 (delta^2-<delta^2>)
        if 'delta2' in field:
            Pk_data /= 2.
        if 'delta2' in field2:
            Pk_data /= 2.

        # those are negative so we make them positive in order to show them in logpsace
        if ((field == 'delta' or field == '1cb') and field2 == 'tidal2') or ((field2 == 'delta' or field2 == '1cb') and field == 'tidal2'):
            Pk_data *= -1 
            Pk_theo *= -1

        # this term is positive if nabla^2 delta = -k^2 delta, but reason we multiply here is that we use k^2 delta instead and k^2 P_zeldovich
        """
        if (field == 'nabla2' and field2 == 'tidal2') or (field2 == 'nabla2' and field == 'tidal2'):
            Pk_data *= -1
        """
        D = 1.#51.77
        if "delta" == field and "delta" == field2:
            Pk_data /= D**2
        elif "delta" == field or "delta" == field2:
            Pk_data /= D
        if "delta2" == field and "delta2" == field2:
            Pk_data /= D**2
        elif "delta2" == field or "delta2" == field2:
            Pk_data /= D
        if "nabla2" == field and "nabla2" == field2:
            Pk_data /= D**2
        elif "nabla2" == field or "nabla2" == field2:
            Pk_data /= D
        if "tidal2" == field and "tidal2" == field2:
            Pk_data /= D**2
        elif "tidal2" == field or "tidal2" == field2:
            Pk_data /= D


        plt.plot(kv, np.abs(Pk_theo), ls='--', color=hexcols[count]) # s del
        plt.plot(k_mid, np.abs(Pk_data), color=hexcols[count], label=f"{field},{field2}")

        plt.xscale('log')
        # plt.yscale('log')
        plt.legend(frameon=False, loc='best')
        count += 1

plt.tight_layout()
plt.savefig("../figures/comparison_MTNG_offset_z0.png")
