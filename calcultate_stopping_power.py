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






def make_energy_hist(E_mean, E_unc, foil, N_bins=200): 

    E_distr = np.random.normal(loc=E_mean, scale=E_unc, size=1000000)
    hist, bin_edges = np.histogram(E_distr, N_bins)

    # plt.hist(E_distr, N_bins, label = foil, color = 'hotpink')
    # plt.xlabel('E (MeV)')
    # plt.legend()
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



def stopping_power_whole_stack(foil_list, rho_ds_list, E_mean_list, E_unc_list):

    stopping_powers_Ti_to_Cu = []
    mean_E_Ti_bins = []
    mean_stopping_power_Ti_to_Cu = []
    mean_E_Ti = []
    stopping_powers_Cu_to_Ti = []
    mean_E_Cu_bins = []
    mean_stopping_power_Cu_to_Ti = []
    mean_E_Cu = []

    for i in range(0, len(E_mean_list), 2): #Calculate all the stopping powers in the Ti foils (energy loss from Ti to Cu, no Fe in between)
        j = 2*i

        hist1, bin_edges1 = make_energy_hist(E_mean_list[i], E_unc_list[i], foil_list[i])
        hist2, bin_edges2 = make_energy_hist(E_mean_list[i+1], E_unc_list[i+1], foil_list[i+1])
        stopping_power, mean_E = stopping_power_bins(hist1, hist2, bin_edges1, bin_edges2, rho_ds_list[j])

        stopping_powers_Ti_to_Cu.append(stopping_power)
        mean_E_Ti_bins.append(mean_E)
        mean_stopping_power_Ti_to_Cu.append(np.mean(stopping_power))
        mean_E_Ti.append(np.mean(mean_E))


    for i, j in zip(range (1, len(E_mean_list)-1, 2), range(1, len(foil_list), 4)): #Calculate the stopping power from Ti to Cu (Al and Fe in between the foils)
        rho_ds = np.sum(rho_ds_list[j:j+3]) #sum rho ds for Cu, Al and Fe

        hist1, bin_edges1 = make_energy_hist(E_mean_list[i], E_unc_list[i], foil_list[i])
        hist2, bin_edges2 = make_energy_hist(E_mean_list[i+1], E_unc_list[i+1], foil_list[i+1])
        stopping_power, mean_E = stopping_power_bins(hist1, hist2, bin_edges1, bin_edges2, rho_ds)

        stopping_powers_Cu_to_Ti.append(stopping_power)
        mean_E_Cu_bins.append(mean_E)
        mean_stopping_power_Cu_to_Ti.append(np.mean(stopping_power))
        mean_E_Cu.append(np.mean(mean_E))
    
    return stopping_powers_Ti_to_Cu, mean_E_Ti_bins, mean_stopping_power_Ti_to_Cu, mean_E_Ti, stopping_powers_Cu_to_Ti, mean_E_Cu_bins, mean_stopping_power_Cu_to_Ti, mean_E_Cu 






if __name__=='__main__':

    # # 25 MeV stack:
    # foil_list = ['Ti02', 'Cu02']
    # E_mean_list = [19.70, 18.86] #[MeV]
    # E_unc_list = [0.095, 0.07] #[MeV]
    # rho_ds_Ti02 = 4.5e3*23.81e-4 #The thickness of the Ti02 foil, [mg/cm^-2]
    # rho_ds_Cu02 = 8.96e3*21.81e-4 #The thickness of the Cu02 foil, [mg/cm^-2]
    # rho_ds_Ti02_Cu02 = 0.5*rho_ds_Ti02 + 0.5*rho_ds_Cu02 #The thickness of half of the Ti02 foil pluss half of the Cu02 foil, [mg/cm^-2]


    # hist1, bin_edges1 = make_energy_hist(E_mean_list[0], E_unc_list[0], foil_list[0])
    # hist2, bin_edges2 = make_energy_hist(E_mean_list[1], E_unc_list[1], foil_list[1])


    # stopping_power_Ti02, mean_E_Ti02 = stopping_power_bins(hist1, hist2, bin_edges1, bin_edges2, rho_ds_Ti02)
    # stopping_power_Ti02_Cu02, mean_E_Ti02_Cu02 = stopping_power_bins(hist1, hist2, bin_edges1, bin_edges2, rho_ds_Ti02_Cu02)

    # colors = ['hotpink', 'orange', 'gold', 'limegreen', 'deepskyblue', 'orchid']

    # plt.plot(mean_E_Ti02, stopping_power_Ti02, color = colors[0], label = 'ds_Ti02')
    # plt.plot(mean_E_Ti02_Cu02, stopping_power_Ti02_Cu02, color = colors[1], label = 'ds_Ti02_Cu02 ')
    # plt.xlabel('E (MeV)')
    # plt.ylabel(r'-dE/$\rho$ds (MeV cm$^2$ mg$^{-1}$)')
    # plt.legend()
    # plt.title('Stopping power of the bins in the Ti02 foil')
    # plt.show()




    # 55 MeV stack with the new function:
    foil_list_55MeV = ['Ti01*', 'Cu01*', 'Al-A1', 'Fe02', 'Ti02', 'Cu02*', 'Al-A2', 'Fe03', 'Ti03', 'Cu03', 'Al-C1', 'Fe04', 'Ti04', 'Cu04', 'Al-C2', 'Fe05', 'Ti05', 'Cu05']
    rho_ds_list_55MeV = [25.88e-4*4.5e3, 28.81e-4*8.83e3, 2.24e-1*2.7e3, 25.5e-4*7.2e3,     25.74e-4*4.5e3, 28.75e-4*8.83e3, 2.24e-1*2.7e3, 25.25e-4*7.2e3,                           25.91e-4*4.5e3, 28.86e-4*8.83e3, 0.97e-1*2.7e3, 25.25e-4*7.2e3,     25.84e-4*4.5e3, 28.78e-4*8.83e3, 0.97e-1*2.7e3, 25.64e-4*7.2e3,                           25.86e-4*4.5e3, 28.77e-4*8.83e3] #The thickness of the foils times the density [mg/cm^-2]

    E_mean_list_55MeV = [53.31, 53.04, 46.61, 46.18, 39.86, 37.60, 34.90, 34.16, 29.86, 28.87] #[MeV]
    E_unc_list_55MeV = [0.61, 0.61, 0.6, 0.68, 0.49, 1.22, 0.22, 0.465, 0.195, 0.155] #[MeV]


    stopping_powers_Ti_to_Cu, mean_E_Ti_bins, mean_stopping_power_Ti_to_Cu, mean_E_Ti, stopping_powers_Cu_to_Ti, mean_E_Cu_bins, mean_stopping_power_Cu_to_Ti, mean_E_Cu = stopping_power_whole_stack(foil_list_55MeV, rho_ds_list_55MeV, E_mean_list_55MeV, E_unc_list_55MeV)

    colors = ['hotpink', 'orange', 'gold', 'limegreen', 'deepskyblue', 'orchid']

    plt.subplot(1,2,1)
    for i, j in zip(range(len(stopping_powers_Ti_to_Cu)),range(0,len(foil_list_55MeV),4)):
        plt.plot(mean_E_Ti_bins[i], stopping_powers_Ti_to_Cu[i], label = f'{foil_list_55MeV[j]}', color = colors[i])
    plt.plot(mean_E_Ti, mean_stopping_power_Ti_to_Cu, 'mo', label = r'mean -dE/$\rho$ds and E in each foil')
    plt.xlabel('E (MeV)')
    plt.ylabel(r'-dE/$\rho$ds (MeV cm$^2$ mg$^{-1}$)')
    plt.legend()
    plt.title('Stopping power of the Ti foils in the 55 MeV stack')

    plt.subplot(1,2,2)
    for i, j in zip(range(len(stopping_powers_Cu_to_Ti)),range(1,len(foil_list_55MeV),4)):
        plt.plot(mean_E_Cu_bins[i], stopping_powers_Cu_to_Ti[i], label = f'{foil_list_55MeV[j]}', color = colors[i])
    plt.plot(mean_E_Cu, mean_stopping_power_Cu_to_Ti, 'mo', label = r'mean -dE/$\rho$ds and E in each foil')
    plt.xlabel('E (MeV)')
    plt.ylabel(r'-dE/$\rho$ds (MeV cm$^2$ mg$^{-1}$)')
    plt.legend()
    plt.title('Stopping power of the Cu, Al and Fe foils in the 55 MeV stack')

    plt.show()








    
