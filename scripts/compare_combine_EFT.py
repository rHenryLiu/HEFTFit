import argparse
import gc
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

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

t0 = time.time()
# Configs
print('Load Configs', time.time()-t0)
heft_dir = '/pscratch/sd/r/rhliu/projects/heft_scratch/'
fig_path = '../figures/good/'
kcut = 0. # for linear power spectrum for velocileptors?
z_mock = 0.0 # config['sim_params']['z_mock']
z_str = '0.54' # for loading the camels pkratios
z_str = '0.00' # for loading the camels pkratios
str_i = 8
str_i = str(str_i)

paste = "TSC"
pcle_type = "A"
factors_fields = {'delta': 1., 'delta2': 2., 'nabla2': 1., 'tidal2': 2}

nmesh = 1080
# nmesh = 1620
nmesh_str = '_' + str(nmesh) + '_'
Lbox = 500
z_ic = 63.
h = 67.76/100
nyquist_freq = np.pi / Lbox * nmesh

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
nmesh_ngen = 1080
n4 = nmesh_ngen//4
density_ngen = dens.copy()
for k in range(nmesh_ngen):
    # print(k)
    density_2d_ngen = density_ngen[:, :, k]

    # shifts up by L/4 in y
    tmp1 = density_2d_ngen[:(nmesh_ngen - n4), :]
    tmp2 = density_2d_ngen[(nmesh_ngen - n4):, :]
    density_2d_ngen[n4:, :] = tmp1
    density_2d_ngen[:n4, :] = tmp2
    density_ngen[:, :, k] = density_2d_ngen
dens = density_ngen.copy()

d, d2, s2, n2 = get_fields(dens, Lbox, nmesh_ngen)
table = {}
table['delta'] = d
table['delta2'] = d2
table['nabla2'] = n2
table['tidal2'] = s2

# Now we calculate the advected fields.
# load our fields first
print('Load Fields', time.time()-t0)

path = '/pscratch/sd/r/rhliu/projects/heft_scratch/MillenniumTNG_sims/'
path_tau = '/pscratch/sd/r/rhliu/projects/heft_scratch/MillenniumTNG_sims/'
# path_tau = '/pscratch/sd/r/rhliu/projects/heft_scratch/MillenniumTNG_sims/fields_3d_1620/'

tau = load_tau(z_mock, path=path_tau)
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
lagr_ijk = ((lagr_pos)/(Lbox/nmesh_ngen)).astype(int)%nmesh_ngen # better than without L/2
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
                 s2_dm_advected, nabla2_dm_advected, delta_tau, Lbox=Lbox, nmesh=nmesh, kmax=nyquist_freq, npoints=101, logscale=False)

del delta_dm_advected, delta_dm_squared_advected, s2_dm_advected, nabla2_dm_advected
gc.collect()

dict_list = []
options = ['field-level-brute', 'field-level-scale', 'field-level-matrix', 'power-spectrum']
options = ['field-level-scale', 'field-level-matrix', 'power-spectrum']
for option in options:
    
    # print(option)
    dict_i = Fit_fn.fit(option, kmax=nyquist_freq, save=False, return_val=True, nbins=41)
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

GaussianProcess = np.load(path_to_files + 'GP_post_fit_MillenniumTNG_singlefield_LH_1080_'+str(z_mock) + '0'+'_0.npy')
k_TNG, PkRatios_TNG = getPkRatios('IllustrisTNG', 'g', 'c', z_str)
# Create a boolean mask for k values <= 10
mask_TNG = k_TNG[0] <= 10
# Apply the mask to filter k values and P(k) values
filtered_k_TNG = k_TNG[:, mask_TNG]
filtered_PkRatios_ITNG = PkRatios_TNG[:, mask_TNG]
filtered_k = filtered_k_TNG[0]

# density_z0 = np.load(path + 'density_mesh__264_MTNG-L500-1080.npy')
# print(np.sum(np.abs(density_z0 - ones_dm_advected)))

# mesh = ArrayMesh(density_z0, BoxSize=[500]*3)
mesh = ArrayMesh(ones_dm_advected, BoxSize=[500]*3)
# delta_tau_mesh = ArrayMesh(delta_tau, BoxSize=[500]*3)

# r1 = FFTPower(mesh, mode='1d', kmax=10)
# Pk1 = r1.power['power'].real[1:]
# r2 = FFTPower(mesh2, mode='1d', kmax=10)
# Pk2 = r2.power['power'].real[1:]
# k = r2.power['k'][1:]

median_Tk = np.sqrt(GaussianProcess)
kk = filtered_k_TNG[0]
print('make and apply transfer fn')
transfer_fn = interpolate.interp1d(kk, median_Tk, bounds_error=False, fill_value='extrapolate')
def transfer(k, v):
    # print(k)
    kk = np.sqrt(sum(ki ** 2 for ki in k))
    tt = transfer_fn(kk)
    return v * tt
field_dm = mesh.to_field(mode='complex')
transfer_field_g = field_dm.apply(transfer)
delta_g = transfer_field_g.c2r()

del mesh, field_dm, transfer_field_g
gc.collect()
print('compute cross_corr')
k_tf, cross_corr = make_cross_corr2(delta_tau, np.array(delta_g), kmax=nyquist_freq)

# P_g1 = FFTPower(delta_tau_mesh, mode='1d', kmax=10)
# P_g2 = FFTPower(delta_g, mode='1d', kmax=10)
# P_cross = FFTPower(delta_g, second=delta_tau_mesh, mode='1d', kmax=10)

# Pk_g1 = P_g1.power['power'].real[1:]
# Pk_g2 = P_g2.power['power'].real[1:]
# Pk_cross = P_cross.power['power'].real[1:]
# kk = P_g2.power['k'][1:]

# cross_corr = P_cross.power['power'].real[1:] /np.sqrt(P_g1.power['power'].real[1:]*P_g2.power['power'].real[1:])


################################################################################

fig = plt.figure(figsize=(8,8))

plt.title(f"r_cc, z = {z_mock:.1f}", fontsize=20)
for j, option in enumerate(options):
    if option == 'power-spectrum':
        
        r_pk = dict_list[j]['r_pk']
        plt.plot(k_avg[:, 0], r_pk[:, 0], label='Power Spectrum Fit')
    if option == 'field-level-matrix':
#         bias = dict_list[j]['one_bias']
#         field_LPT = Fit_fn.get_field(bias)
#         k, r_pk = make_cross_corr2(delta_tau, field_LPT)
        
#         plt.plot(k, r_pk, label='Fixed Bias HEFT')
#         del field_LPT
#         gc.collect()
#         continue
        
        r_pk = dict_list[j]['r_pk']
        plt.plot(k_avg[:, 0], r_pk[:, 0], label='Fixed Bias HEFT')
    if option == 'field-level-scale':
#         bias_k = dict_list[0]['one_bias']
#         kbins_k = dict_list[0]['kbins']
#         field_LPT = Fit_fn.get_field(bias_k, kbins=kbins_k, scale=True)
#         k, r_pk = make_cross_corr2(delta_tau, field_LPT)
        
#         plt.plot(k, r_pk, label='Varying Bias HEFT')
#         del field_LPT
#         gc.collect()

        r_pk = dict_list[j]['r_pk']
        plt.plot(k_avg[:, 0], r_pk[:, 0], label='Varying Bias HEFT')
    else:
        continue
# ax0.plot(k_avg[:, i], r_pk[:, i], label="field-level-brute")
# ax0.plot(k_avg[:, i], r_pk_bk[:, i], label="field-level-scale")
# ax0.plot(k_avg[:, i], r_pk_alt[:, i], label="field-level-matrix")
# plt.plot(k_avg[:, i], r_pk_fit[:, i], label="power-spectrum")
# plt.axvline(5, c='k', label='Nyquist Frequency Cutoff')
plt.plot(k_tf, cross_corr, label='Eulerian Transfer Functon')
plt.legend(fontsize=14)
plt.xscale('log')
plt.xlabel('k')
plt.ylabel('r(k)')
plt.tight_layout()
plt.savefig(fig_path + "r_cc_compare_z_" + str(z_mock) + nmesh_str + str_i + ".png", dpi=100)
plt.close()

################################################################################
k_dm, r_pk2 = make_cross_corr2(ones_dm_advected, delta_tau, kmax=nyquist_freq) # cross corr coeff with DM
kk, Pk_g2 = calc_power(np.array(delta_g), kmax=nyquist_freq)
kk, Pcross = calc_power(np.array(delta_g), field2=delta_tau, kmax=nyquist_freq)
Pk_error_tf = Pk_g2 - 2*Pcross + Fit_fn.pk_tau

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for i in range(1):

    # plt.figure(1)
    ax0 = ax[0]
    ax0.set_title(f"r_cc, z = {z_mock:.1f}")
    ax0.plot(k_dm, r_pk2, label='DM')
    for j, option in enumerate(options):
        # if option == 'power-spectrum':
        #     continue
        r_pk = dict_list[j]['r_pk']
        ax0.plot(k_avg[:, i], r_pk[:, i], label=option)

    ax0.plot(k_tf, cross_corr, label='Eulerian Transfer Functon')
    ax0.legend()
    ax0.set_xscale('log')
    ax0.set_xlabel('k')
    ax0.set_ylabel('r(k)')
    # plt.ylim([0, 1.0]) 

    # plt.figure(2)
    ax1 = ax[1]
    ax1.set_title(f"Power Spectrum, z = {z_mock:.1f}")
    for j, option in enumerate(options):
        pk_mod = dict_list[j]['pk_mod']
        ax1.plot(k_avg[:, i], pk_mod[:, i]*k_avg[:, i]**3/2./np.pi**2, label=option)

    ax1.errorbar(k_avg[:, i], pk_tau[:, i]*k_avg[:, i]**3/2./np.pi**2, yerr=np.sqrt(2./Fit_fn.Nmode[:, i])*pk_tau[:, i]*k_avg[:, i]**3/2./np.pi**2, capsize=4, label="Tau")
    pk_dm = Fit_fn.power_dict['ones_dm_adv_ones_dm_adv']
    ax1.errorbar(k_avg[:, i], pk_dm[:, i]*k_avg[:, i]**3/2./np.pi**2, yerr=np.sqrt(2./Fit_fn.Nmode[:, i])*pk_dm[:, i]*k_avg[:, i]**3/2./np.pi**2, capsize=4, label="DM")

    ax1.plot(kk, Pk_g2*kk**3/2./np.pi**2, lw=3, label='Transfer Functon')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('k')
    ax1.set_ylabel('P(k)')

    # plt.figure(3)
    ax2 = ax[2]
    ax2.set_title(f"Pk error ratio, z = {z_mock:.1f}")
    for j, option in enumerate(options):
        pk_err = dict_list[j]['pk_err']
        ax2.plot(k_avg[:, i], np.abs(pk_err/pk_tau)[:, i], label=option)
        
    ax2.plot(kk, (Pk_error_tf/pk_tau)
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('k')
    # plt.ylim([0, 0.5]) 
    ax2.set_ylabel('P_err(k)/P_tau(k)')
plt.tight_layout()
plt.savefig(fig_path + "output_plots_z_" + str(z_mock) + nmesh_str + str_i + ".png", dpi=100)
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

kk, r_LPT = make_cross_corr2(field_halo, field_LPT, kmax=nyquist_freq)
kk, r_EPT = make_cross_corr2(field_halo, field_EPT, kmax=nyquist_freq)
kk, r_true = make_cross_corr2(field_halo, field_tau, kmax=nyquist_freq)

fig = plt.figure(figsize=(8,8))
plt.plot(kk, r_LPT, label='r_cc for Halos and LPT reconstructed tau Field')
plt.plot(kk, r_EPT, label='r_cc for Halos and EPT reconstructed tau Field')
plt.plot(kk, r_true, label='r_cc for Halos and true tau Field')
plt.xscale('log')
plt.legend()
plt.savefig(fig_path + "Halo_r_cc_compare_z_" + str(z_mock) + nmesh_str + str_i + ".png", dpi=100)
# plt.savefig('../figures/Halo_r_cc.png', dpi=100)

print('Done!!! Total runtime:', time.time()-t0)