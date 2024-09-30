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

import matplotlib
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import json
import pprint

t0 = time.time()

# Configs

print('Load Config from JSON', time.time()-t0)
# JSON Parameters:
param_Dict = json.loads(sys.argv[1])
locals().update(param_Dict)

print('Parameters:')
pprint.pprint(param_Dict)

################################################################################
'''
Params:
z_mock = 0.0, 0.5, 1.0
fig_path = "../figures/good"
Y_compton = true or false

nmesh = 1080 or 1620 (technically 2160 is possible but computationally infeasible)
Lbox = 500
Halo_mass_threshold = 1e14
fit_amplitude = true or false
'''

if z_mock == 0.0:
    z_str = '0.00'
elif z_mock == 0.5:
    z_str = '0.54'
elif z_mock == 1.0:
    z_str = '1.05'
else:
    raise ValueError('Invalid z_mock value')


# fig_path = '../figures/good/'
# z_mock = 0.0 # config['sim_params']['z_mock']
# z_str = '0.54' # for loading the camels pkratios
# z_str = '0.00' # for loading the camels pkratios

factors_fields = {'delta': 1., 'delta2': 2., 'nabla2': 1., 'tidal2': 2}

# nmesh = 1080
# nmesh = 1620
# Lbox = 500

nmesh_str = '_' + str(nmesh) + '_'
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

# VERY IMPORTANT LINE: position offsets
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

if (not Y_compton):
    tau = load_tau(z_mock, path=path_tau)
else:
    tau = load_Y_compton(z_mock, path=path_tau)
    
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


lagr_ijk = ((lagr_pos)/(Lbox/nmesh_ngen)).astype(int)%nmesh_ngen # Correct lagrangian positions 
for field in factors_fields.keys():
    w = (table[field]*D_ratio**factors_fields[field])[lagr_ijk[:,0], lagr_ijk[:,1], lagr_ijk[:,2]]

    tsc_parallel(pcle_pos, adv_fields[field], Lbox, weights=w)
tsc_parallel(pcle_pos, adv_fields['1cb'], Lbox, weights=None)


print('now for fitting tau', time.time()-t0)

# load our advected fields - HL

ones_dm_advected = adv_fields['1cb']
delta_dm_advected = adv_fields['delta']
delta_dm_squared_advected = adv_fields['delta2']
s2_dm_advected = adv_fields['tidal2']
nabla2_dm_advected = adv_fields['nabla2']

delta_tau = tau/np.mean(tau) - 1

# normalize 

normalize_mean = np.mean(ones_dm_advected, dtype=np.float64)
ones_dm_advected /= normalize_mean
ones_dm_advected -= 1.
delta_dm_advected /= normalize_mean
delta_dm_squared_advected /= normalize_mean
s2_dm_advected /= normalize_mean
nabla2_dm_advected /= normalize_mean

# Initialize HEFTFit class

Fit_fn = HEFTFit(ones_dm_advected, delta_dm_advected, delta_dm_squared_advected, 
                 s2_dm_advected, nabla2_dm_advected, delta_tau, Lbox=Lbox, 
                 nmesh=nmesh, kmax=nyquist_freq, npoints=41, logscale=False)

del delta_dm_advected, delta_dm_squared_advected, s2_dm_advected, nabla2_dm_advected
gc.collect()

dict_list = []
# options = ['field-level-brute', 'field-level-scale', 'field-level-matrix', 'power-spectrum']
options = ['field-level-scale', 'field-level-matrix']
for option in options:
    
    # print(option)
    dict_i = Fit_fn.fit(option, kmax=nyquist_freq, save=False, return_val=True, nbins=41, fit_amplitude=fit_amplitude)
    dict_list.append(dict_i)

print(dict_list[0].keys())

k_avg = dict_list[0]['k_avg']
pk_tau = Fit_fn.pk_tau

################################################################################
print('Now for the Transfer Function Method', time.time()-t0)

if (not Y_compton):
    # If Y_compton is false, then we load the Gaussian Processes for the Tau fit
    # instead of just making direct transfer functions like the Y compton case.

    path_to_files = '/pscratch/sd/r/rhliu/projects/heft_transfer_fn/MillenniumTNG/GP_test_outputs/'

    GaussianProcess = np.load(path_to_files + 'GP_post_fit_MillenniumTNG_singlefield2_LH_1080_'+str(z_mock) + '0'+'_0.npy')
    k_TNG, PkRatios_TNG = getPkRatios('IllustrisTNG', 'g', 'c', z_str)

    # Create a boolean mask for k values <= 10
    mask_TNG = k_TNG[0] <= 10
    # Apply the mask to filter k values and P(k) values
    filtered_k_TNG = k_TNG[:, mask_TNG]
    filtered_PkRatios_ITNG = PkRatios_TNG[:, mask_TNG]
    filtered_k = filtered_k_TNG[0]
    
    Tk = np.sqrt(GaussianProcess)
    kk = filtered_k_TNG[0]
    transfer_fn = interpolate.interp1d(kk, Tk, bounds_error=False, fill_value='extrapolate')
    
#     k_tf, Pk1 = calc_power(ones_dm_advected, kmax=nyquist_freq)
#     k_tf, Pk2 = calc_power(delta_tau, kmax=nyquist_freq)
#     # Tk = np.sqrt(Pk2/Pk1)
    
#     plt.semilogx(k_tf, Pk2/Pk1, label='Pk Ratio')
#     plt.semilogx(kk, GaussianProcess, label='GP')
#     plt.legend()
#     plt.savefig(fig_path + 'TF_comparison.png', dpi=100)
else:
    # If Y_compton is true, then we're considering the actual Y_compton parameter
    # instead of basing our work on the gaussian process emulator, in this case
    # we're computing the actual transfer function rather than the emulator transfer
    # function.
    
#     mesh = ArrayMesh(ones_dm_advected, BoxSize=[500]*3)
#     mesh2 = ArrayMesh(delta_tau, BoxSize=[500]*3)
    
#     r1 = FFTPower(mesh, mode='1d', kmax=nyquist_freq)
#     Pk1 = r1.power['power'].real[1:]
#     r2 = FFTPower(mesh2, mode='1d', kmax=nyquist_freq)
#     Pk2 = r2.power['power'].real[1:]
#     kk = r2.power['k'][1:]
    
    kk, Pk1 = calc_power(ones_dm_advected, kmax=nyquist_freq)
    kk, Pk2 = calc_power(delta_tau, kmax=nyquist_freq)
    Tk = np.sqrt(Pk2/Pk1)
    
    transfer_fn = interpolate.interp1d(kk, Tk, bounds_error=False, fill_value='extrapolate')

# Apply transfer function to dm (1cb) field:

mesh = ArrayMesh(ones_dm_advected, BoxSize=[500]*3)
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


################################################################################
# Now for plotting our field accuracy metrics
################################################################################

################################################################################
k_dm, r_pk2 = make_cross_corr2(ones_dm_advected, delta_tau, kmax=nyquist_freq) # cross corr coeff with DM

kk, Pk_g2 = calc_power2(np.array(delta_g), k_bin_edges=Fit_fn.k_bin_edges)
kk, Pcross = calc_power2(np.array(delta_g), field2=delta_tau, k_bin_edges=Fit_fn.k_bin_edges)
kk, PkTau = calc_power2(delta_tau, k_bin_edges=Fit_fn.k_bin_edges)
Pk_error_tf = Pk_g2 - 2*Pcross + PkTau

if (not Y_compton):
    label = 'Tau'
    savestr = 'tau'
else:
    label = 'Y Compton'
    savestr = 'Y_comp'


cmap = matplotlib.cm.get_cmap('tab10')
cmap = matplotlib.cm.get_cmap('viridis')
colours = cmap(np.linspace(0, 1, 10))
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for i in range(1):

    # plt.figure(1)
    ax0 = ax[0]
    ax0.set_title(f"r_cc with " + label + f", z = {z_mock:.1f}")
    # ax0.plot(k_dm, r_pk2, label='DM')
    for j, option in enumerate(options):
        # if option == 'power-spectrum':
        #     continue
        r_pk = dict_list[j]['r_pk']
        ax0.plot(k_avg[:, i], r_pk[:, i], label=option, c=colours[j])

    ax0.plot(k_tf, cross_corr, label='Eulerian Transfer Functon', c=colours[2])
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
        ax1.plot(k_avg[:, i], pk_mod[:, i]*k_avg[:, i]**3/2./np.pi**2, label=option, c=colours[j])

    ax1.errorbar(k_avg[:, i], pk_tau[:, i]*k_avg[:, i]**3/2./np.pi**2, yerr=np.sqrt(2./Fit_fn.Nmode[:, i])*pk_tau[:, i]*k_avg[:, i]**3/2./np.pi**2, capsize=4, label=label, c=colours[3])
    pk_dm = Fit_fn.power_dict['ones_dm_adv_ones_dm_adv']
    ax1.errorbar(k_avg[:, i], pk_dm[:, i]*k_avg[:, i]**3/2./np.pi**2, yerr=np.sqrt(2./Fit_fn.Nmode[:, i])*pk_dm[:, i]*k_avg[:, i]**3/2./np.pi**2, capsize=4, label="DM", c=colours[4])

    ax1.plot(kk, Pk_g2*kk**3/2./np.pi**2, lw=3, label='Transfer Functon', c=colours[2])
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
        ax2.plot(k_avg[:, i], np.abs(pk_err/pk_tau)[:, i], label=option, c=colours[j])
        
    ax2.plot(kk, np.abs(Pk_error_tf/pk_tau.flatten()), label='Tranfer Function', c=colours[2])
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('k')
    # plt.ylim([0, 0.5]) 
    ax2.set_ylabel('P_err(k)/P_tau(k)')
plt.tight_layout()
print('Plot saving to: '+ fig_path + "output_plots_z_" + str(z_mock) + nmesh_str + savestr + ".png")
plt.savefig(fig_path + "output_plots_z_" + str(z_mock) + nmesh_str + savestr + ".png", dpi=100)
plt.close()

################################################################################
# Now for cross correlations with Group Haloes
print('Plot Halo Cross Correlation Metrics', time.time() - t0)
path_halo = '/pscratch/sd/r/rhliu/projects/heft_scratch/MTNG_haloes/'
GroupPos = load_GroupPos(z_mock, path_halo)
GroupMass = load_GroupMass(z_mock, path_halo) * 1e10

Halo_mass_thresholds = [1e12, 1e13, 1e14]
Halo_mass_thresholds_str = ['1e12', '1e13', '1e14']


fig, ax = plt.subplots(3, 3, figsize=(20, 20))
for i, Halo_mass_threshold in enumerate(Halo_mass_thresholds):

    mask = GroupMass >= Halo_mass_threshold
    GroupPos_masked = GroupPos[mask]
    GroupMass_masked = GroupMass[mask]


    # field_halo = tsc_parallel(GroupPos_masked, (nmesh, nmesh, nmesh), Lbox, weights=GroupMass_masked)
    field_halo = tsc_parallel(GroupPos_masked, (nmesh, nmesh, nmesh), Lbox)
    field_halo = field_halo / np.mean(field_halo) - 1

    matrix_bias = dict_list[1]['one_bias']
    field_LPT_m = Fit_fn.get_field(matrix_bias)
    bias_k = dict_list[0]['one_bias']
    kbins_k = dict_list[0]['kbins']
    field_LPT_s = Fit_fn.get_field(bias_k, kbins=kbins_k, scale=True)

    field_EPT = np.array(delta_g)
    field_tau = delta_tau.copy()

    kk, r_LPT_m = make_cross_corr2(field_halo, field_LPT_m, kmax=nyquist_freq)
    kk, r_LPT_s = make_cross_corr2(field_halo, field_LPT_s, kmax=nyquist_freq)
    kk, r_EPT = make_cross_corr2(field_halo, field_EPT, kmax=nyquist_freq)
    kk, r_true = make_cross_corr2(field_halo, field_tau, kmax=nyquist_freq)

    kk, Pk_halo = calc_power(field_halo, kmax=nyquist_freq)
    kk, Pk_tau = calc_power(field_tau, kmax=nyquist_freq)
    kk, Pk_EPT = calc_power(field_EPT, kmax=nyquist_freq)
    kk, Pk_LPT_m = calc_power(field_LPT_m, kmax=nyquist_freq)
    kk, Pk_LPT_s = calc_power(field_LPT_s, kmax=nyquist_freq)

    kk, Pkx_LPT_m = calc_power(field_halo, field2=field_LPT_m, kmax=nyquist_freq)
    kk, Pkx_LPT_s = calc_power(field_halo, field2=field_LPT_s, kmax=nyquist_freq)
    kk, Pkx_EPT = calc_power(field_halo, field2=field_EPT, kmax=nyquist_freq)
    kk, Pkx_true = calc_power(field_halo, field2=field_tau, kmax=nyquist_freq)


    ax[i][0].set_title(f"Halo r_cc, \n z = {z_mock:.1f},"+r" Halo Mass $\geq$" + Halo_mass_thresholds_str[i])
    ax[i][0].plot(kk, r_LPT_m, label='r_cc for Halos and scale-independent LPT (' + label +')')
    ax[i][0].plot(kk, r_LPT_s, label='r_cc for Halos and scale-dependent LPT (' + label +')')
    ax[i][0].plot(kk, r_EPT, label='r_cc for Halos and TF ( ' + label +')')
    ax[i][0].plot(kk, r_true, label='r_cc for Halos and true (' + label +')')
    ax[i][0].set_xscale('log')


    ax[i][1].set_title(f"Halo Cross Power Spectra, \n z = {z_mock:.1f},"+r" Halo Mass $\geq$" + Halo_mass_thresholds_str[i])   
    ax[i][1].plot(kk, Pkx_LPT_m, label='Pkx of Halo and LPT (matrix)')
    ax[i][1].plot(kk, Pkx_LPT_s, label='Pkx of Halo and LPT (scale)')
    ax[i][1].plot(kk, Pkx_EPT, label='Pkx of Halo and TF')
    ax[i][1].plot(kk, Pkx_true, label='Pkx of Halos and true (' + label + ')')
    ax[i][1].set_xscale('log')
    ax[i][1].set_yscale('log')
    

    ax[i][2].set_title(r"$\frac{P_{x, reconstructed}}{P_{x, true}}$, " + "\n " + f"z = {z_mock:.1f},"+r" Halo Mass $\geq$" + Halo_mass_thresholds_str[i])    
    ax[i][2].plot(kk, np.abs(Pkx_LPT_m/Pkx_true) - 1, label='LPT (Matrix)')
    ax[i][2].plot(kk, np.abs(Pkx_LPT_s/Pkx_true) - 1, label='LPT (Scale)')
    ax[i][2].plot(kk, np.abs(Pkx_EPT/Pkx_true) - 1, label='TF')
    ax[i][2].set_xscale('log')

    # ax[i][2].set_title(f"Halo Power Spectra, \n z = {z_mock:.1f},"+r" Halo Mass $\geq$" + Halo_mass_thresholds_str[i])    
    # ax[i][2].plot(kk, Pk_halo, label='Pk of halo')
    # ax[i][2].plot(kk, Pk_tau, label='Pk of ' + label)
    # ax[i][2].plot(kk, Pk_EPT, ls=':', label='Pk from TF')
    # ax[i][2].plot(kk, Pk_LPT, c='b', label='Pk from LPT')
    # ax[i][2].set_xscale('log')
    # ax[i][2].set_yscale('log')
    
    if i==2:
        ax[i][0].legend(fontsize=11)
        ax[i][1].legend(fontsize=11)
        ax[i][2].legend(fontsize=11)

plt.tight_layout()
print('Plot saving to: '+ fig_path + "Halo_r_cc_compare_z_" + str(z_mock) + nmesh_str + savestr + ".png")
plt.savefig(fig_path + "Halo_r_cc_compare_z_" + str(z_mock) + nmesh_str + savestr + ".png", dpi=100)


print('Done!!! Total runtime:', time.time()-t0)