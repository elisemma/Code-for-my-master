from statistics import variance
import curie as ci
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import glob
from scipy.interpolate import CubicSpline


def energy_in_target_Hannah(E0, dp):

    st = ci.Stack('stackedTarget_193mPt.csv', E0=E0, N=100000, particle='p', dp = dp)
    st.saveas(f'hannahs_stack{E0:.3f}MeV{dp:.3f}.csv')



def extract_data_Hannah(filename, targets):

    csv_stack_data = pd.read_csv(f'{filename}.csv')
    csv_flux_data = pd.read_csv(f'{filename}_fluxes.csv')

    areal_dens_list = []
    averageE_list = []
    sigE_list = []
    E_list = []
    flux_list = []

    for target in targets:
        target_stack_data = csv_stack_data.loc[csv_stack_data['name'] == target] 
        target_flux_data = csv_flux_data.loc[csv_flux_data['name'] == target] 

        areal_dens = target_stack_data.loc[:,'areal_density']
        energy_average = target_stack_data.loc[:,'mu_E']
        energy_sigma = target_stack_data.loc[:, 'sig_E']
        energy = target_flux_data.loc[:,'energy']
        flux = target_flux_data.loc[:,'flux']

        areal_dens_converted = areal_dens.values.tolist()
        energy_average_converted = energy_average.values.tolist()
        energy_sigma_converted = energy_sigma.values.tolist()
        energy_converted = energy.values.tolist()
        flux_converted = flux.values.tolist()

        areal_dens_list.append(float(areal_dens_converted[0])) #[mg/cm^2]
        averageE_list.append(energy_average_converted) #[MeV]
        sigE_list.append(energy_sigma_converted) #[MeV]
        E_list.append(energy_converted) #[MeV]
        flux_list.append(flux_converted)

    return areal_dens_list, averageE_list, sigE_list, E_list, flux_list



def plot_mean_FWHM_Hannah(filename, targets):

    areal_dens_list, averageE_list, sigE_list, E_list, flux_list = extract_data_Hannah(filename, targets) 
    flux_list[-1][:2] = [0,0]
    flux_list[-2][:2] = [0,0]


    print('_____________________________FWHM________________________________')
    for i in range(len(E_list)):
        flux_max = np.max(flux_list[i])
        flux_max_index = np.argmax(flux_list[i])

        upper_half_flux_max_index = flux_max_index + np.argmin(abs((flux_max/2) - flux_list[i][flux_max_index:-1]))
        upper_half_flux_max = flux_list[i][upper_half_flux_max_index]
      
        lower_half_flux_max_index = np.argmin(abs((flux_max/2) - flux_list[i][0:flux_max_index]))
        lower_half_flux_max = flux_list[i][lower_half_flux_max_index]

        fwhm_rhs_x_list = [E_list[i][flux_max_index], E_list[i][upper_half_flux_max_index]]
        fwhm_lhs_x_list = [E_list[i][lower_half_flux_max_index], E_list[i][flux_max_index]]
        fwhm_y_list = [flux_max/2, flux_max/2]

        fwhm_rhs = E_list[i][upper_half_flux_max_index]-E_list[i][flux_max_index]
        fwhm_lhs = E_list[i][flux_max_index] - E_list[i][lower_half_flux_max_index]

        print(f'For target number {10-i} the FWHM is {fwhm_lhs :.3f} MeV for the left hand side and {fwhm_rhs :.3f} MeV for the right hand side.')

        plt.plot(fwhm_lhs_x_list, fwhm_y_list, color = 'lightcoral')
        plt.plot(fwhm_rhs_x_list, fwhm_y_list, color = 'lightgreen')
        plt.plot(E_list[i], flux_list[i], label = f'Target number {i}')
        plt.axvline(x = E_list[i][flux_max_index], linestyle = '--', color = 'grey')
    plt.title(f'E0={E0}MeV, dp={dp}')
    plt.legend()    
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Flux (a.u)")    
    plt.show()








if __name__ == '__main__':
    E0 = 33.0 #[MeV]
    dp = 1.0
    targets = ['Ir0' + str(i) for i in range(1,10)]
    targets.append('Ir10')
    filename = f'hannahs_stack{E0:.3f}MeV{dp:.3f}'

    energy_in_target_Hannah(E0, dp)
    plot_mean_FWHM_Hannah(filename, targets)
    







