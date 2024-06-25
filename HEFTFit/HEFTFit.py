import argparse
import gc
import os
from pathlib import Path
import warnings
import time
import sys

import sys
# import asdf
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

# import field_utils
# sys.path.append('.')
from HEFTFit.field_utils import *

# from asdf.exceptions import AsdfWarning
# warnings.filterwarnings('ignore', category=AsdfWarning)

sys.path.append('../velocileptors') # clone the velocileptors from github 
from velocileptors.LPT.cleft_fftw import CLEFT



class HEFTFit(object):
    def __init__(self, ones_dm_adv, delta_dm_adv, delta_dm_squared_adv, 
                  s2_dm_adv, nabla2_dm_adv, delta_tau, Lbox, nmesh, ):
        
        self.Lbox = Lbox
        self.nmesh = nmesh
        
        self.delta_tau_obs_fft = (rfftn(delta_tau, workers=-1) 
                                  / np.complex64(delta_tau.size))

        # adv stands for advected
        self.ones_dm_adv_fft = (rfftn(ones_dm_adv, workers=-1) 
                                / np.complex64(ones_dm_adv.size))
        self.delta_dm_adv_fft = (rfftn(delta_dm_adv, workers=-1) 
                                 / np.complex64(delta_dm_adv.size))
        self.delta_dm_squared_adv_fft = (rfftn(delta_dm_squared_adv, workers=-1)
                                         / np.complex64(delta_dm_squared_adv.size))
        self.s2_dm_adv_fft = (rfftn(s2_dm_adv, workers=-1) 
                              / np.complex64(s2_dm_adv.size))
        self.nabla2_dm_adv_fft = (rfftn(nabla2_dm_adv, workers=-1) 
                                  / np.complex64(nabla2_dm_adv.size))
        
        
        self.k_bin_edges = np.linspace(1e-2, 1., 21) # best
        self.k_binc = (self.k_bin_edges[1:]+self.k_bin_edges[:-1])*.5
        self.mu_bin_edges = np.array([0, 1.])
        self.mu_binc = (self.mu_bin_edges[1:]+self.mu_bin_edges[:-1])*.5
        
        kx, ky, kz = fftfreq(nmesh, d=Lbox/nmesh)*2.*np.pi, fftfreq(nmesh, d=Lbox/nmesh)*2.*np.pi, rfftfreq(nmesh, d=Lbox/nmesh)*2.*np.pi
        kx, ky, kz = np.meshgrid(kx, ky, kz)#, indexing='ij') # TESTING
        self.k2 = kx**2 + ky**2 + kz**2
        self.mu2 = kz**2/self.k2
        self.mu2[0, 0, 0] = 0.
        
        
        result = calc_pk_from_deltak(self.delta_tau_obs_fft, self.Lbox, self.k_bin_edges, self.mu_bin_edges)

        pk = result['power']
        Nmode = result['N_mode']
        binned_poles = result['binned_poles']
        N_mode_poles = result['N_mode_poles']
        k_avg = result['k_avg']
        if len(self.mu_binc) == 1:
            self.pk_tau = np.atleast_2d(pk).T
            self.k_avg = np.atleast_2d(k_avg).T
            self.Nmode = np.atleast_2d(Nmode).T
        else:
            self.pk_tau = pk

        self.get_power_dict()
            
    
    def fit(self, option, kmin=0.0, kmax=0.4, mumax=1.01, save=True, return_val=False):
        
        print(option)
        
        if option == 'field-level-brute':
        # Field-level fit
            def mini_fun(bias):
                b1, b2, bs, bn = np.real(bias)
                delta_model_fft_cut = ones_dm_adv_fft_cut + b1*delta_dm_adv_fft_cut + b2*delta_dm_squared_adv_fft_cut + bs*s2_dm_adv_fft_cut + bn*nabla2_dm_adv_fft_cut
                diff = delta_tau_obs_fft_cut - delta_model_fft_cut 
                sum_diff2 = np.sum(diff*np.conj(diff))
                return sum_diff2
        
            threshold = (self.k2 < kmax**2) & (self.k2 > kmin**2) & (self.mu2 < mumax**2)
            N_points = np.sum(threshold)
            print(N_points)
            assert N_points != 0
            
            delta_tau_obs_fft_cut = self.delta_tau_obs_fft[threshold]
            ones_dm_adv_fft_cut = self.ones_dm_adv_fft[threshold]
            delta_dm_adv_fft_cut = self.delta_dm_adv_fft[threshold]
            delta_dm_squared_adv_fft_cut = self.delta_dm_squared_adv_fft[threshold]
            nabla2_dm_adv_fft_cut = self.nabla2_dm_adv_fft[threshold]
            s2_dm_adv_fft_cut = self.s2_dm_adv_fft[threshold]
            #self.k2_cut = self.k2[(self.k2 < kmax**2) & (self.k2 > kmin**2)]

            x0 = [1., 1., 1., 1.] # initial guess for the bias parameters
            res = minimize(mini_fun, x0, method='Powell')
            b1, b2, bs, bn = np.real(res['x'])
            # print(b1, b2, bs, bn)
            one_bias = np.array([1., b1, b2, bs, bn])
            print(one_bias)

            # construct model (only need if looking at other statistics)
            #delta_model_fft = ones_dm_adv_fft + b1*delta_dm_adv_fft + b2*delta_dm_squared_adv_fft + bs*s2_dm_adv_fft + bn*nabla2_dm_adv_fft

            pk_mod = self.get_power_model(one_bias, kmin=0., kmax=np.inf, mumax=np.inf)
            pk_tau_mod = self.get_cross_power_model(one_bias, kmin=0., kmax=np.inf, mumax=np.inf)
            r_pk = pk_tau_mod/(np.sqrt(self.pk_tau*pk_mod)) # cross corr coeff - HL
            pk_err = pk_mod - 2*pk_tau_mod + self.pk_tau

            if return_val:
                return_dict = dict(k_avg=self.k_avg, Nmode=self.Nmode, 
                                   pk_tau=self.pk_tau, pk_tau_mod=pk_tau_mod, pk_mod=pk_mod, 
                                   pk_err=pk_err, r_pk=r_pk, one_bias=one_bias)
                return return_dict
            if save:
                np.savez(f"data/pk_bias.npz", k_avg=self.k_avg, Nmode=self.Nmode, 
                         pk_tau=self.pk_tau, pk_tau_mod=pk_tau_mod, pk_mod=pk_mod, 
                         pk_err=pk_err, r_pk=r_pk, one_bias=one_bias)
        
        elif option == 'field-level-matrix':
            
            threshold = (self.k2 < kmax**2) & (self.k2 > kmin**2) & (self.mu2 < mumax**2)
            N_points = np.sum(threshold)
            print(N_points)
            assert N_points != 0
            
            delta_tau_obs_fft_cut = self.delta_tau_obs_fft[threshold]
            ones_dm_adv_fft_cut = self.ones_dm_adv_fft[threshold]
            delta_dm_adv_fft_cut = self.delta_dm_adv_fft[threshold]
            delta_dm_squared_adv_fft_cut = self.delta_dm_squared_adv_fft[threshold]
            nabla2_dm_adv_fft_cut = self.nabla2_dm_adv_fft[threshold]
            s2_dm_adv_fft_cut = self.s2_dm_adv_fft[threshold]
            #self.k2_cut = self.k2[(self.k2 < kmax**2) & (self.k2 > kmin**2)]


            operators = ["delta_dm_adv", "delta_dm_squared_adv", "s2_dm_adv", "nabla2_dm_adv"]
            A_dict = {}
            M_dict = {}
            for i, operator_i in enumerate(operators):
                A_dict[operator_i] = np.sum(locals()[f"{operator_i}_fft_cut"]*np.conj(delta_tau_obs_fft_cut - ones_dm_adv_fft_cut))
                for j, operator_j in enumerate(operators):
                    if i < j: continue
                    M_dict[f"{operator_i}_{operator_j}"] = np.sum(locals()[f"{operator_i}_fft_cut"]*np.conj(locals()[f"{operator_j}_fft_cut"]))
                    M_dict[f"{operator_j}_{operator_i}"] = np.conj(M_dict[f"{operator_i}_{operator_j}"])

            A_vect = np.zeros(len(operators), dtype=np.cdouble)
            M_matr = np.zeros((len(operators), len(operators)), dtype=np.cdouble)
            for i in range(len(operators)):
                A_vect[i] = A_dict[operators[i]]
                for j in range(len(operators)):
                    M_matr[i, j] = M_dict[f"{operators[i]}_{operators[j]}"]
            one_bias_alt = np.hstack((1., np.dot(np.linalg.inv(M_matr), A_vect).real))
            print(one_bias_alt)

            pk_mod_alt = self.get_power_model(one_bias_alt, kmin=0., kmax=np.inf)
            pk_tau_mod_alt = self.get_cross_power_model(one_bias_alt, kmin=0., kmax=np.inf)
            r_pk_alt = pk_tau_mod_alt/(np.sqrt(self.pk_tau*pk_mod_alt))
            pk_err_alt = pk_mod_alt - 2*pk_tau_mod_alt + self.pk_tau

            # TESTING
            #pk_mod = pk_mod_alt
            #r_pk = r_pk_alt
            
            if return_val:
                return_dict = dict(k_avg=self.k_avg, Nmode=self.Nmode, pk_tau=self.pk_tau, 
                                   pk_tau_mod=pk_tau_mod_alt, pk_mod=pk_mod_alt, 
                                   pk_err=pk_err_alt, r_pk=r_pk_alt, one_bias=one_bias_alt)
                return return_dict
            
            if save:
                np.savez(f"data/pk_alt_bias.npz", k_avg=self.k_avg, Nmode=self.Nmode, 
                         pk_tau=self.pk_tau, pk_tau_mod=pk_tau_mod_alt, pk_mod=pk_mod_alt, 
                         pk_err=pk_err_alt, r_pk=r_pk_alt, one_bias=one_bias_alt)

        elif option == 'field-level-scale':
            # Fits for scale-dependent bias
            kbins = np.linspace(0., 1., 21) # best
            # kbins = np.linspace(kmin, kmax, 21) # best

            operators = ["delta_dm_adv", "delta_dm_squared_adv", "s2_dm_adv", "nabla2_dm_adv"]
            for k in range(len(kbins)-1):
                kmax = kbins[k+1]
                kmin = kbins[k]
                mumax = 1.01 # 0.1
                
                vals_dict = self.fit('field-level-matrix', kmin=kmin, kmax=kmax, mumax=mumax, 
                                     save=False, return_val=True)
                one_bias = vals_dict['one_bias']
                
                # print(one_bias)
                if k == 0:
                    one_bias_bk = one_bias
                else:
                    one_bias_bk = np.vstack((one_bias_bk, one_bias[:1+len(operators)]))
            gc.collect()
            # plot bias super lazy
            kbinc = (kbins[1:]+kbins[:-1])*.5
            # for i in range(1, one_bias_bk.shape[1]):
            #     plt.figure(i)
            #     plt.plot(kbinc, one_bias_bk[:, i])
            #     plt.savefig(f"figs/one_bias_bk_{i:d}.png")
            #     plt.close()

            # get power spectrum
            pk_mod_bk = self.get_power_model(np.hstack((one_bias_bk[2], np.zeros(5 - len(one_bias_bk[2])))), 
                                             kmin=0., kmax=np.inf)
            pk_tau_mod_bk = self.get_cross_power_model(np.hstack((one_bias_bk[2], np.zeros(5 - len(one_bias_bk[2])))),
                                                      kmin=0., kmax=np.inf)
            #r_pk_bk = pk_tau_mod_bk/(np.sqrt(pk_tau*pk_mod_bk))
            #pk_err_bk = pk_mod_bk - 2*pk_tau_mod_bk + pk_tau

            # get error power
            for k in range(len(kbins)-1):
                kmax = kbins[k+1]
                kmin = kbins[k]
                pk_mod_bk[(kmin < self.k_avg) & (self.k_avg <= kmax)] = self.get_power_model(np.hstack((one_bias_bk[k], np.zeros(5 - len(one_bias_bk[k])))), kmin=0., kmax=np.inf)[(kmin < self.k_avg) & (self.k_avg <= kmax)]
                pk_tau_mod_bk[(kmin < self.k_avg) & (self.k_avg <= kmax)] = self.get_cross_power_model(np.hstack((one_bias_bk[k], np.zeros(5 - len(one_bias_bk[k])))), kmin=0., kmax=np.inf)[(kmin < self.k_avg) & (self.k_avg <= kmax)]
            r_pk_bk = pk_tau_mod_bk/(np.sqrt(self.pk_tau*pk_mod_bk))
            pk_err_bk = pk_mod_bk - 2*pk_tau_mod_bk + self.pk_tau

            if return_val:
                return_dict = dict(k_avg=self.k_avg, Nmode=self.Nmode, 
                         pk_tau=self.pk_tau, pk_tau_mod=pk_tau_mod_bk, pk_mod=pk_mod_bk, 
                         pk_err=pk_err_bk, r_pk=r_pk_bk, one_bias=one_bias_bk, 
                         kbinc=kbinc)
                return return_dict
            if save:
                np.savez(f"data/pk_bk_bias.npz", k_avg=self.k_avg, Nmode=self.Nmode, 
                         pk_tau=self.pk_tau, pk_tau_mod=pk_tau_mod_bk, pk_mod=pk_mod_bk, 
                         pk_err=pk_err_bk, r_pk=r_pk_bk, one_bias=one_bias_bk, 
                         kbinc=kbinc)
        
        elif option == 'power-spectrum':
            # Power-spectrum fit
            
            def mini_fun_pk(bias, kmin=kmin, kmax=kmax, mumax=mumax):
                one_bias = np.hstack(([1.], bias))
                pk_mod_cut = self.get_power_model(one_bias, kmin, kmax, mumax)
                pk_tau_cut = self.pk_tau[((self.k_binc < kmax) & (self.k_binc > kmin))[:, None] & (self.mu_binc < mumax)[None, :]].reshape(np.sum(((self.k_binc < kmax) & (self.k_binc > kmin))), np.sum(self.mu_binc < mumax))
                sum_diff2 = np.sum((pk_tau_cut-pk_mod_cut)**2/pk_tau_cut**2)
                return sum_diff2

            
            x0 = [1., 1., 1., 1.] # initial guess for the bias parameters
            N_points = np.sum(((self.k_binc < kmax) & (self.k_binc > kmin))) * np.sum(self.mu_binc < mumax)
            print(N_points)
            assert N_points != 0
            res = minimize(mini_fun_pk, x0, args=(kmin, kmax, mumax), method='Powell')
            b1_fit, b2_fit, bs_fit, bn_fit = res['x']
            print(b1_fit, b2_fit, bs_fit, bn_fit)
            one_bias_fit = np.array([1., b1_fit, b2_fit, bs_fit, bn_fit])
            # print(one_bias_fit)

            # get error power
            pk_mod_fit = self.get_power_model(one_bias_fit, kmin=0., kmax=np.inf, mumax=np.inf)
            pk_tau_mod_fit = self.get_cross_power_model(one_bias_fit, kmin=0., kmax=np.inf, mumax=np.inf)
            r_pk_fit = pk_tau_mod_fit/(np.sqrt(self.pk_tau*pk_mod_fit))
            pk_err_fit = pk_mod_fit - 2*pk_tau_mod_fit + self.pk_tau

            if return_val:
                return_dict = dict(k_avg=self.k_avg, Nmode=self.Nmode, pk_tau=self.pk_tau, 
                                   pk_tau_mod=pk_tau_mod_fit, pk_mod=pk_mod_fit, 
                                   pk_err=pk_err_fit, r_pk=r_pk_fit, 
                                   one_bias=one_bias_fit)
                return return_dict
            
            if save:
                np.savez(f"data/pk_fit_bias.npz", k_avg=self.k_avg, 
                         Nmode=self.Nmode, pk_tau=self.pk_tau, pk_tau_mod=pk_tau_mod_fit, 
                         pk_mod=pk_mod_fit, pk_err=pk_err_fit, r_pk=r_pk_fit, 
                         one_bias=one_bias_fit)
         

        else:
            raise ValueError('option is not one of the possible fit options, choose between '
                             + 'field-level-brute, field-level-matrix, field-level-scale, '
                             + 'or power-spectrum')
                              

    def get_power_model(self, one_bias, kmin=0., kmax=np.inf, mumax=np.inf):
        power_model = np.zeros((np.sum((self.k_binc <= kmax) & (self.k_binc > kmin)), np.sum(self.mu_binc < mumax)))
        fields = ["ones_dm_adv", "delta_dm_adv", "delta_dm_squared_adv", "s2_dm_adv", "nabla2_dm_adv"]
        for i, field_i in enumerate(fields):
            for j, field_j in enumerate(fields):
                power_model += one_bias[i] * one_bias[j] * self.power_dict[f"{field_i}_{field_j}"][((self.k_binc <= kmax) & (self.k_binc > kmin))[:, None] & (self.mu_binc < mumax)[None, :]].reshape(np.sum(((self.k_binc <= kmax) & (self.k_binc > kmin))), np.sum(self.mu_binc < mumax))
        return power_model

    def get_cross_power_model(self, one_bias, kmin=0., kmax=np.inf, mumax=np.inf):
        power_model = np.zeros((np.sum((self.k_binc <= kmax) & (self.k_binc > kmin)), np.sum(self.mu_binc < mumax)))
        fields = ["ones_dm_adv", "delta_dm_adv", "delta_dm_squared_adv", "s2_dm_adv", "nabla2_dm_adv"]
        for i, field_i in enumerate(fields):
            power_model += one_bias[i] * self.power_dict[f"delta_tau_obs_{field_i}"][((self.k_binc <= kmax) & (self.k_binc > kmin))[:, None] & (self.mu_binc < mumax)[None, :]].reshape(np.sum(((self.k_binc <= kmax) & (self.k_binc > kmin))), np.sum(self.mu_binc < mumax))
        return power_model

    def get_power_dict(self):
        
        power_dict = {}
        fields = ["ones_dm_adv", "delta_dm_adv", "delta_dm_squared_adv", "s2_dm_adv", "nabla2_dm_adv"]
        print(self.__dict__.keys())
        for i, field_i in enumerate(fields):
            result = calc_pk_from_deltak(self.delta_tau_obs_fft, self.Lbox, self.k_bin_edges, self.mu_bin_edges, field2_fft=self.__dict__[f"{field_i}_fft"])

            pk = result['power']
            Nmode = result['N_mode']
            binned_poles = result['binned_poles']
            N_mode_poles = result['N_mode_poles']
            k_avg = result['k_avg']   
            if len(self.mu_binc) == 1:
                pk = np.atleast_2d(pk).T
                k_avg = np.atleast_2d(k_avg).T
                Nmode = np.atleast_2d(Nmode).T
            power_dict[f"delta_tau_obs_{field_i}"] = pk

            for j, field_j in enumerate(fields):
                if i < j: continue
                result = calc_pk_from_deltak(self.__dict__[f"{field_i}_fft"], self.Lbox, self.k_bin_edges, self.mu_bin_edges, field2_fft=self.__dict__[f"{field_j}_fft"])

                pk = result['power']
                Nmode = result['N_mode']
                binned_poles = result['binned_poles']
                N_mode_poles = result['N_mode_poles']
                k_avg = result['k_avg']
                if len(self.mu_binc) == 1:
                    pk = np.atleast_2d(pk).T
                    k_avg = np.atleast_2d(k_avg).T
                    Nmode = np.atleast_2d(Nmode).T
                power_dict[f"{field_i}_{field_j}"] = power_dict[f"{field_j}_{field_i}"] = pk

        self.power_dict = power_dict