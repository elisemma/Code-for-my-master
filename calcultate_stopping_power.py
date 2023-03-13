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

    mean_bin_E1_list = []
    mean_bin_E2_list = []

    for i in range(len(hist1)):
        E1 = (bin_edges1[i] + bin_edges1[i+1])/2
        E2 = (bin_edges2[i] + bin_edges2[i+1])/2
        mean_bin_E1_list.append(E1)
        mean_bin_E2_list.append(E2)

    mean_bin_E1 = np.array(mean_bin_E1_list)
    mean_bin_E2 = np.array(mean_bin_E2_list)

    dE = mean_bin_E2 - mean_bin_E1

    return -dE/ds,  mean_bin_E1 #returning array of stopping power for all the bins and an array with the mean energies of the bins in the first foil



def stopping_power_whole_stack(foil_list, rho_ds_list, E_mean_list, E_unc_list):

    stopping_powers_Ti_to_Cu = []
    mean_E_Ti_bins = []
    mean_stopping_power_Ti_to_Cu = []
    mean_E_Ti = []
    stopping_powers_Cu_to_Ti = []
    mean_E_Cu_bins = []
    mean_stopping_power_Cu_to_Ti = []
    mean_E_Cu = []

    for i, j in zip(range(0, len(E_mean_list), 2), range(0, len(foil_list), 4)): #Calculate all the stopping powers in the Ti foils (energy loss from Ti to Cu, no Fe in between)
        rho_ds = 0.5*rho_ds_list[j]+0.5*rho_ds_list[j+1]

        hist1, bin_edges1 = make_energy_hist(E_mean_list[i], E_unc_list[i], foil_list[i])
        hist2, bin_edges2 = make_energy_hist(E_mean_list[i+1], E_unc_list[i+1], foil_list[i+1])
        stopping_power, mean_bin_E = stopping_power_bins(hist1, hist2, bin_edges1, bin_edges2, rho_ds)

        stopping_powers_Ti_to_Cu.append(stopping_power)
        mean_E_Ti_bins.append(mean_bin_E)

        mean_stopping_power_Ti_to_Cu.append((E_mean_list_55MeV[i]-E_mean_list_55MeV[i+1])/(rho_ds))
        mean_E_Ti.append(E_mean_list[i])
    print(mean_stopping_power_Ti_to_Cu)


    for i, j in zip(range (1, len(E_mean_list)-1, 2), range(1, len(foil_list), 4)): #Calculate the stopping power from Ti to Cu (Al and Fe in between the foils)
        rho_ds = 0.5*rho_ds_list[j] + np.sum(rho_ds_list[j+1:j+3]) + 0.5*rho_ds_list[j+3] #sum rho ds for Cu, Al and Fe

        hist1, bin_edges1 = make_energy_hist(E_mean_list[i], E_unc_list[i], foil_list[i])
        hist2, bin_edges2 = make_energy_hist(E_mean_list[i+1], E_unc_list[i+1], foil_list[i+1])
        stopping_power, mean_bin_E = stopping_power_bins(hist1, hist2, bin_edges1, bin_edges2, rho_ds)

        stopping_powers_Cu_to_Ti.append(stopping_power)
        mean_E_Cu_bins.append(mean_bin_E)

        mean_stopping_power_Cu_to_Ti.append((E_mean_list_55MeV[i]-E_mean_list_55MeV[i+1])/(rho_ds))
        mean_E_Cu.append(np.mean(mean_bin_E))
    print(mean_stopping_power_Cu_to_Ti)

    
    return stopping_powers_Ti_to_Cu, mean_E_Ti_bins, mean_stopping_power_Ti_to_Cu, mean_E_Ti, stopping_powers_Cu_to_Ti, mean_E_Cu_bins, mean_stopping_power_Cu_to_Ti, mean_E_Cu 




def curie_stopping_power(element, energy):
     el = ci.Element(element)
     S_curie = el.S(energy, density=1E-3)
     return S_curie





if __name__=='__main__':

    energy = np.linspace(0,60, 100000)

    S_Cu = curie_stopping_power('Cu', energy)
    S_Al = curie_stopping_power('Al', energy)
    S_Ti = curie_stopping_power('Ti', energy)
    S_Fe = curie_stopping_power('Fe', energy)

    # plt.plot(energy, S_Cu, label = 'Cu', color = 'hotpink')
    # plt.plot(energy, S_Al, label = 'Al', color = 'deepskyblue')
    # plt.plot(energy, S_Ti, label = 'Ti', color = 'gold')
    # plt.plot(energy, S_Fe, label = 'Fe', color = 'lightgreen')
    # plt.legend()
    # plt.xlabel('Energy (Mev)')
    # plt.ylabel('Mass stopping power (MeV/(mg/cm^2)')
    # plt.show()




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
    # rho_ds_list_55MeV = [11.09, 22.40, xxx, 19.91, 10.94, 22.32, xxx, 20.00, 11.25, 22.49, xxx, 19.93, 10.91, 22.38, xxx, 20.02, 10.99, 22.35] #Areal density [mg/cm^-2]
    # rho_ds_unc_list_55MeV = [0.16, 0.11, xxx, 0.13, 0.24, 0.40, xxx, 0.27, 0.15, 0.20, xxx, 0.33, 0.18, 0.29, xxx, 0.24, 0.30, 0.12] #Uncertainty in percent 

    rho_ds_list_55MeV = [11.09, 22.40, 0.224*2.7, 19.91, 10.94, 22.32, 0.224*2.7, 20.00, 11.25, 22.49, 0.097*2.7, 19.93, 10.91, 22.38, 0.097*2.7, 20.02, 10.99, 22.35] #Areal density [mg/cm^-2]
    rho_ds_unc_list_55MeV = [0.16, 0.11, 0.2, 0.13, 0.24, 0.40, 0.2, 0.27, 0.15, 0.20, 0.2, 0.33, 0.18, 0.29, 0.2, 0.24, 0.30, 0.12] #Uncertainty in percent 

    E_mean_list_55MeV = [53.31, 53.04, 46.61, 46.18, 39.86, 37.60, 34.90, 34.16, 29.86, 28.87] #[MeV]
    E_unc_list_55MeV = [0.61, 0.61, 0.6, 0.68, 0.49, 1.22, 0.22, 0.465, 0.195, 0.155] #[MeV]


    stopping_powers_Ti_to_Cu, mean_E_Ti_bins, mean_stopping_power_Ti_to_Cu, mean_E_Ti, stopping_powers_Cu_to_Ti, mean_E_Cu_bins, mean_stopping_power_Cu_to_Ti, mean_E_Cu = stopping_power_whole_stack(foil_list_55MeV, rho_ds_list_55MeV, E_mean_list_55MeV, E_unc_list_55MeV)



    colors = ['hotpink', 'orange', 'gold', 'limegreen', 'deepskyblue', 'orchid']

    plt.subplot(1,2,1)
    plt.plot(energy, S_Cu, label = 'Cu, Curie', color = 'black')
    plt.plot(energy, S_Al, label = 'Al, Curie', color = 'dimgray')
    plt.plot(energy, S_Ti, label = 'Ti, Curie', color = 'darkgrey')
    plt.plot(energy, S_Fe, label = 'Fe, Curie', color = 'lightgrey')
    for i, j in zip(range(len(stopping_powers_Ti_to_Cu)),range(0,len(foil_list_55MeV),4)):
        plt.plot(mean_E_Ti_bins[i], stopping_powers_Ti_to_Cu[i], label = f'{foil_list_55MeV[j]}', color = colors[i])
    plt.plot(mean_E_Ti, mean_stopping_power_Ti_to_Cu, 'mo', label = r'mean -dE/$\rho$ds and E in each foil')
    plt.xlabel('E (MeV)')
    plt.ylabel(r'-dE/$\rho$ds (MeV cm$^2$ mg$^{-1}$)')
    plt.legend()
    plt.title('Stopping power of the Ti foils in the 55 MeV stack')

    plt.subplot(1,2,2)
    plt.plot(energy, S_Cu, label = 'Cu, Curie', color = 'black')
    plt.plot(energy, S_Al, label = 'Al, Curie', color = 'dimgray')
    plt.plot(energy, S_Ti, label = 'Ti, Curie', color = 'darkgrey')
    plt.plot(energy, S_Fe, label = 'Fe, Curie', color = 'lightgrey')
    for i, j in zip(range(len(stopping_powers_Cu_to_Ti)),range(1,len(foil_list_55MeV),4)):
        plt.plot(mean_E_Cu_bins[i], stopping_powers_Cu_to_Ti[i], label = f'{foil_list_55MeV[j]}', color = colors[i])
    plt.plot(mean_E_Cu, mean_stopping_power_Cu_to_Ti, 'mo', label = r'mean -dE/$\rho$ds and E in each foil')
    plt.xlabel('E (MeV)')
    plt.ylabel(r'-dE/$\rho$ds (MeV cm$^2$ mg$^{-1}$)')
    plt.legend()
    plt.title('Stopping power of the Cu, Al and Fe foils in the 55 MeV stack')

    plt.show()








    
