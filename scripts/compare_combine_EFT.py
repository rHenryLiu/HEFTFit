import argparse
from pathlib import Path
import warnings
import time

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq
from scipy.optimize import minimize
from scipy import interpolate
import scipy as sp

from classy import Class
import abacusnbody
import abacusnbody.analysis
from abacusnbody.analysis.tsc import tsc_parallel #put it on a grid using tsc interpolation
from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak #computes power spectrum from density contrast, not specific to abacus 
# from abacusnbody.analysis.power_spectrum import index_3d_rfft

# from obtain_IC_fields import *

sys.path.append('../velocileptors') # clone the velocileptors from github 
from velocileptors.LPT.cleft_fftw import CLEFT
sys.path.append('../') # Add the HEFTFit package
from HEFTFit.field_utils import *
from HEFTFit.HEFTFit import HEFTFit
sys.path.append('../../transfer_fcn/scripts')
from emulator_utils import *
from nbodykit.lab import *
import h5py
import hdf5plugin

t0 = time.time()
# Configs
print('Load Configs', time.time()-t0)
heft_dir = '/pscratch/sd/r/rhliu/projects/heft_scratch/'
nmesh = 1080
kcut = 0. # for linear power spectrum for velocileptors?
z_mock = 0.0 # config['sim_params']['z_mock']

paste = "TSC"
pcle_type = "A"
factors_fields = {'delta': 1., 'delta2': 2., 'nabla2': 1., 'tidal2': 2}

nmesh = 1080
Lbox = 500
z_ic = 63.
h = 67.76/100

# Making growth factor for MTNG using CLASS
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
D_ratio = D_mock/D_ic # Growth ratio

# load fields

path = '/pscratch/sd/r/rhliu/projects/heft_scratch/MillenniumTNG_sims/'

# Load Lagrangian Density first
dens = np.load(path + 'density_ngenic.npy')

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

# Now we calculate the advected fields.
# load our fields first
print('Load Fields', time.time()-t0)

path = '/pscratch/sd/r/rhliu/projects/heft_scratch/MillenniumTNG_sims/'

tau = load_tau(z_mock, path)
lagr_pos = load_lagrangians(path=path)
pcle_pos = load_positions(z=z_mock, path=path)
velocity = load_velocities(z=z_mock, path=path)

# VERY IMPORTANT LINE: position offsets
# Either do this for these fields or for the IC field (dens) above, better to do 
# it above for dens I believe.

# lagr_pos[:, 0] -= Lbox/4
# lagr_pos = lagr_pos % Lbox
# pcle_pos[:, 0] -= Lbox/4
# pcle_pos = pcle_pos % Lbox

# Then calculate advected fields:
print('Calculate Advected Fields', time.time()-t0)
adv_fields = {}
for field in factors_fields.keys():
    adv_fields[field] = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
adv_fields['1cb'] = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)


# lagr_ijk = ((lagr_pos+Lbox/2.)/(Lbox/nmesh)).astype(int)%nmesh # better than without L/2
lagr_ijk = ((lagr_pos)/(Lbox/nmesh)).astype(int)%nmesh # better than without L/2
for field in factors_fields.keys():
    # w = (f['data'][field]*D_ratio**factors_fields[field])[lagr_ijk[:,0], lagr_ijk[:,1], lagr_ijk[:,2]]
    w = (table[field]*D_ratio**factors_fields[field])[lagr_ijk[:,0], lagr_ijk[:,1], lagr_ijk[:,2]]

    if paste == "TSC":
        tsc_parallel(pcle_pos, adv_fields[field], Lbox, weights=w)

if paste == "TSC":
    tsc_parallel(pcle_pos, adv_fields['1cb'], Lbox, weights=None)


print('now for fitting tau', time.time()-t0)

# load advected
# our advected fields - HL

ones_dm_advected = adv_fields['1cb']
delta_dm_advected = adv_fields['delta']
delta_dm_squared_advected = adv_fields['delta2']
s2_dm_advected = adv_fields['tidal2']
nabla2_dm_advected = adv_fields['nabla2']

delta_tau = tau/np.mean(tau) - 1 # Question: Is this advected? A: No

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

# Initialize HEFTFit class

Fit_fn = HEFTFit(ones_dm_advected, delta_dm_advected, delta_dm_squared_advected, 
                 s2_dm_advected, nabla2_dm_advected, delta_tau, Lbox=500, nmesh=1080)

dict_list = []
options = ['field-level-brute', 'field-level-scale', 'field-level-matrix', 'power-spectrum']
options = ['field-level-scale', 'field-level-matrix', 'power-spectrum']
for option in options:
    
    # print(option)
    dict_i = Fit_fn.fit(option, kmax=10.0, save=False, return_val=True)
    dict_list.append(dict_i)

print(dict_list[0].keys())

k_avg = dict_list[0]['k_avg']
pk_tau = Fit_fn.pk_tau

################################################################################

print('Now for the Transfer Function Method', time.time()-t0)

path_to_files = '/pscratch/sd/r/rhliu/projects/heft_transfer_fn/GP_test_outputs/'
path_to_files = '/pscratch/sd/r/rhliu/projects/heft_transfer_fn/MillenniumTNG/GP_test_outputs/'
sim_name1 = 'IllustrisTNG_g'
sim_name2 = 'IllustrisTNG_m'

save_path = '/pscratch/sd/r/rhliu/projects/heft_transfer_fn/'

GaussianProcess = np.load(path_to_files + 'GP_post_fit_MillenniumTNG_singlefield_LH_1080_0.00_0.npy')
k_TNG, PkRatios_TNG = getPkRatios('IllustrisTNG', 'g', 'c', '0.00')
# Create a boolean mask for k values <= 10
mask_TNG = k_TNG[0] <= 10
# Apply the mask to filter k values and P(k) values
filtered_k_TNG = k_TNG[:, mask_TNG]
filtered_PkRatios_ITNG = PkRatios_TNG[:, mask_TNG]
filtered_k = filtered_k_TNG[0]

density_z0 = np.load(path + 'density_mesh__264_MTNG-L500-1080.npy')
print(np.sum(np.abs(density_z0 - ones_dm_advected)))

# mesh = ArrayMesh(density_z0, BoxSize=[500]*3)
mesh = ArrayMesh(ones_dm_advected, BoxSize=[500]*3)
delta_tau_mesh = ArrayMesh(delta_tau, BoxSize=[500]*3)

# r1 = FFTPower(mesh, mode='1d', kmax=10)
# Pk1 = r1.power['power'].real[1:]
# r2 = FFTPower(mesh2, mode='1d', kmax=10)
# Pk2 = r2.power['power'].real[1:]
# k = r2.power['k'][1:]

median_Tk = np.sqrt(GaussianProcess)
kk = filtered_k_TNG[0]
transfer_fn = interpolate.interp1d(kk, median_Tk, bounds_error=False, fill_value='extrapolate')
def transfer(k, v):
    # print(k)
    kk = np.sqrt(sum(ki ** 2 for ki in k))
    tt = transfer_fn(kk)
    return v * tt
field_dm = mesh.to_field(mode='complex')
transfer_field_g = field_dm.apply(transfer)
delta_g = transfer_field_g.c2r()

P_g1 = FFTPower(delta_tau_mesh, mode='1d', kmax=10)
P_g2 = FFTPower(delta_g, mode='1d', kmax=10)
P_cross = FFTPower(delta_g, second=delta_tau_mesh, mode='1d', kmax=10)

Pk_g1 = P_g1.power['power'].real[1:]
Pk_g2 = P_g2.power['power'].real[1:]
Pk_cross = P_cross.power['power'].real[1:]
k = P_cross.power['k'][1:]

cross_corr = P_cross.power['power'].real[1:] /np.sqrt(P_g1.power['power'].real[1:]*P_g2.power['power'].real[1:])

################################################################################

fig = plt.figure(figsize=(8,8))

plt.title(f"r_cc, z = {z_mock:.1f}")
for j, option in enumerate(options):
    if option == 'power-spectrum':
        continue
    r_pk = dict_list[j]['r_pk']
    plt.plot(k_avg[:, 0], r_pk[:, 0], label=option)
# ax0.plot(k_avg[:, i], r_pk[:, i], label="field-level-brute")
# ax0.plot(k_avg[:, i], r_pk_bk[:, i], label="field-level-scale")
# ax0.plot(k_avg[:, i], r_pk_alt[:, i], label="field-level-matrix")
# plt.plot(k_avg[:, i], r_pk_fit[:, i], label="power-spectrum")
plt.plot(k, cross_corr, label='transfer_function EPT')
plt.legend()
plt.xscale('log')
plt.xlabel('k')
plt.ylabel('r(k)')
plt.savefig("../HEFTFit/figures/r_cc_compare_z_" + str(z_mock) + ".png")
plt.close()

################################################################################
# Now for cross correlations with Group Haloes
print('Plot Cross Correlations', time.time() - t0)
path_halo = '/pscratch/sd/r/rhliu/projects/heft_scratch/MTNG_haloes/'

GroupPos = load_GroupPos(z_mock, path_halo)
GroupMass = load_GroupMass(z_mock, path_halo) * 1e10

field_halo = tsc_parallel(GroupPos, (nmesh, nmesh, nmesh), Lbox, weights=GroupMass)
print(np.mean(field_halo))
field_halo = field_halo / np.mean(field_halo) - 1

matrix_bias = dict_list[1]['one_bias']
field_LPT = Fit_fn.get_field(matrix_bias)
print(np.mean(field_LPT))
# field_LPT = field_LPT/np.mean(field_LPT) - 1

field_EPT = np.array(delta_g)
print(np.mean(field_EPT))
# field_EPT = field_EPT/np.mean(field_EPT) - 1

field_tau = delta_tau.copy()

kk, r_LPT = make_cross_corr(field_halo, field_LPT)
kk, r_EPT = make_cross_corr(field_halo, field_EPT)
kk, r_true = make_cross_corr(field_halo, field_tau)

fig = plt.figure(figsize=(8,8))
plt.plot(kk, r_LPT, label='r_cc for Halos and LPT reconstructed tau Field')
plt.plot(kk, r_EPT, label='r_cc for Halos and EPT reconstructed tau Field')
plt.plot(kk, r_true, label='r_cc for Halos and true tau Field')
plt.xscale('log')
plt.legend()
plt.savefig("../figures/Halo_r_cc_compare_z_" + str(z_mock) + ".png")
# plt.savefig('../figures/Halo_r_cc.png', dpi=100)

print('Done!!! Total runtime:', time.time()-t0)