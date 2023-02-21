from statistics import variance
import curie as ci
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import glob
from scipy.interpolate import CubicSpline
from scipy.interpolate import splev, splrep
from scipy.interpolate import interp1d




def make_energy_hist(E_mean, E_unc, N_bins=200): 

    E_distr = np.random.normal(loc=E_mean, scale=E_unc, size=1000000)
    hist, bin_edges = np.histogram(E_distr, N_bins)

    # print('hist: ', hist)
    # print('bin edg: ', bin_edges)

    plt.hist(E_distr, N_bins, color = 'pink')
    plt.xlabel('MeV')
    # plt.show()

    return hist, bin_edges



def stopping_power_bins(hist1, hist2, bin_edges1, bin_edges2, ds): #1 is before 2 in the stack and thus has the highest energy 

    mean_E1_list = []
    mean_E2_list = []

    for i in range(len(hist1)):
        E1 = (bin_edges1[i] + bin_edges1[i+1])/2
        E2 = (bin_edges2[i] + bin_edges2[i+1])/2
        mean_E1_list.append(E1)
        mean_E2_list.append(E2)

    mean_E1 = np.array(mean_E1_list)
    mean_E2 = np.array(mean_E2_list)

    dE = mean_E2 - mean_E1

    return -dE/ds,  mean_E1 #returning array of stopping power for all the bins and an array with the mean energies of the bins in the first foil






if __name__=='__main__':

    # foil_list = ['Ti01', 'Cu01', 'Fe02', 'Ti02', 'Cu02', 'Cu01', 'Fe03', 'Ti03', 'Cu03']
    foil_list = ['Ti02', 'Cu02']
    E_mean_list = [19.70, 18.86] #[MeV]
    E_unc_list = [0.095, 0.07] #[MeV]
    rho_ds_Ti02 = 4.5e3*23.81e-4 #The thickness of the Ti02 foil, [mg/cm^-2]
    rho_ds_Cu02 = 8.96e3*21.81e-4 #The thickness of the Cu02 foil, [mg/cm^-2]
    rho_ds_Ti02_Cu02 = 0.5*rho_ds_Ti02 + 0.5*rho_ds_Cu02 #The thickness of half of the Ti02 foil pluss half of the Cu02 foil, [mg/cm^-2]

    E_distr_list = []
    E_bins_list = []
    colors = ['skyblue', 'pink']


    plt.subplot(1,2,1)
    hist1, bin_edges1 = make_energy_hist(E_mean_list[0], E_unc_list[0])
    hist2, bin_edges2 = make_energy_hist(E_mean_list[1], E_unc_list[1])


    stopping_power_Ti02, mean_E_Ti02 = stopping_power_bins(hist1, hist2, bin_edges1, bin_edges2, rho_ds_Ti02)
    stopping_power_Ti02_Cu02, mean_E_Ti02_Cu02 = stopping_power_bins(hist1, hist2, bin_edges1, bin_edges2, rho_ds_Ti02_Cu02)


    plt.subplot(1,2,2)
    plt.plot(mean_E_Ti02, stopping_power_Ti02, color = 'lightskyblue', label = 'ds_Ti02')
    plt.plot(mean_E_Ti02_Cu02, stopping_power_Ti02_Cu02, color = 'pink', label = 'ds_Ti02_Cu02 ')
    plt.xlabel('E (MeV)')
    plt.ylabel(r'-dE/$\rho$ds (MeV cm$^2$ mg$^{-1}$)')
    plt.legend()
    plt.title('Stopping power of the bins in the Ti02 foil')
    plt.show()












    
