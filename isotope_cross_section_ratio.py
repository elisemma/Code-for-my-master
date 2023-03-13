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


def get_xs_from_exfor_files(beam, reaction_list):
    E_list = []
    E_unc_list =[]
    xs_list = []
    xs_unc_list = []

    if beam == 'p':
        path = 'xs_EXFOR_protons_txt/'

    elif beam == 'd':
        path = 'xs_EXFOR_deuterons_txt/'

    else:
        print('Er du sikker på at du ikke har proton eller deuteron beam?')

    for reaction in reaction_list:
        with open(path+reaction+'.txt') as file:

            lines = file.readlines()[11:-2]
            E_reaction_list = []
            E_unc_reaction_list = []
            xs_reaction_list = []
            xs_unc_reaction_list = []

            for line in lines:
                words = line.split()
                E_reaction_list.append(float(words[0]))
                E_unc_reaction_list.append(float(words[1]))
                xs_reaction_list.append(float(words[2]))
                xs_unc_reaction_list.append(float(words[3]))

            E_list.append(E_reaction_list)
            E_unc_list.append(E_unc_reaction_list)
            xs_list.append(xs_reaction_list)
            xs_unc_list.append(xs_unc_reaction_list)
            file.close()

    # return np.array(E_list), np.array(E_unc_list), np.array(xs_list), np.array(xs_unc_list)
    return E_list, E_unc_list, xs_list, xs_unc_list



def generate_xs_lists(beam, reaction_list):
    E_list = []
    xs_list = []
    xs_unc_list = []

    if beam == 'p':
        path = 'xs_IAEA_protons_txt/'

    elif beam == 'd':
        path = 'xs_IAEA_deuterons_txt/'

    else:
        print('Er du sikker på at du ikke har proton eller deuteron beam?')

    for reaction in reaction_list:
        with open(path+reaction+'.txt') as file:
            # file.readline(6)
            lines = file.readlines()[6:]
            E_reaction_list = []
            xs_reaction_list = []
            xs_unc_reaction_list = []

            for line in lines:
                words = line.split()
                E_reaction_list.append(float(words[0]))
                xs_reaction_list.append(float(words[1]))
                xs_unc_reaction_list.append(float(words[2]))

            E_list.append(E_reaction_list)
            xs_list.append(xs_reaction_list)
            xs_unc_list.append(xs_unc_reaction_list)
            file.close()

    return E_list, xs_list, xs_unc_list



def interpol_xs_iaea(xs_list, E_list):

    cs = CubicSpline(E_list, xs_list)

    return cs



def interpol_xs_exfor(xs_list, E_list):
    # spl = splrep(E_list, xs_list, k = 5, s = 1)
    interp = interp1d(E_list, xs_list)

    return interp



def exp_xs_ratio_from_A0(A0, A0_prime, decay_const, decay_const_prime, t_irr):

    ratio = (A0/(1-np.exp(-decay_const*t_irr))) / (A0_prime/(1-np.exp(-decay_const_prime*t_irr)))

    return ratio 



def exp_xs_ratio_from_A0_w_unc(A0, A0_prime, t_half, t_half_prime, t_irr, A0_unc, A0_prime_unc, t_half_unc, t_half_prime_unc, t_irr_unc):

    decay_const = np.log(2)/t_half    
    decay_const_prime = np.log(2)/t_half_prime

    ratio = exp_xs_ratio_from_A0(A0, A0_prime, decay_const, decay_const_prime, t_irr) #Calculating the ratio


    #Starting the uncertainty calculations___________________________________________
    #Uncertainty of the decay constanrs:
    dt_half = t_half*1e-8
    dfdt_half = (np.log(2)/(t_half+dt_half/2) - np.log(2)/(t_half-dt_half/2))/dt_half
    decay_const_unc = np.sqrt(dfdt_half*dfdt_half*t_half_unc*t_half_unc)

    dt_half_prime = t_half_prime*1e-8
    dfdt_half_prime = (np.log(2)/(t_half_prime+dt_half_prime/2) - np.log(2)/(t_half_prime-dt_half_prime/2))/dt_half_prime
    decay_const_prime_unc = np.sqrt(dfdt_half_prime*dfdt_half_prime*t_half_prime_unc*t_half_prime_unc)


    #Uncertainty of the ratio:
    dfdx_list = [] #Jacobian
    unc_list = []

    dA0 = A0*1e-8
    dfdA0 = (exp_xs_ratio_from_A0(A0+dA0/2, A0_prime, decay_const, decay_const_prime, t_irr) - exp_xs_ratio_from_A0(A0-dA0/2, A0_prime, decay_const, decay_const_prime, t_irr))/dA0
    dfdx_list.append(dfdA0)
    unc_list.append(A0_unc)

    dA0_prime = A0_prime*1e-8
    dfdA0_prime = (exp_xs_ratio_from_A0(A0, A0_prime+dA0_prime/2, decay_const, decay_const_prime, t_irr) - exp_xs_ratio_from_A0(A0, A0_prime-dA0_prime/2, decay_const, decay_const_prime, t_irr))/dA0_prime
    dfdx_list.append(dfdA0_prime)
    unc_list.append(A0_prime_unc)

    ddecay_const = decay_const*1e-8
    dfddecay_const = (exp_xs_ratio_from_A0(A0, A0_prime, decay_const+ddecay_const/2, decay_const_prime, t_irr) - exp_xs_ratio_from_A0(A0, A0_prime, decay_const-ddecay_const/2, decay_const_prime, t_irr))/ddecay_const
    dfdx_list.append(dfddecay_const)
    unc_list.append(decay_const_unc)

    ddecay_const_prime = decay_const_prime*1e-8
    dfddecay_const_prime = (exp_xs_ratio_from_A0(A0, A0_prime, decay_const, decay_const_prime+ddecay_const_prime/2, t_irr) - exp_xs_ratio_from_A0(A0, A0_prime, decay_const, decay_const_prime-ddecay_const_prime/2, t_irr))/ddecay_const_prime
    dfdx_list.append(dfddecay_const_prime)
    unc_list.append(decay_const_prime_unc)

    dt_irr = t_irr*1e-8
    dfdt_irr = (exp_xs_ratio_from_A0(A0, A0_prime, decay_const, decay_const_prime, t_irr+dt_irr/2)-exp_xs_ratio_from_A0(A0, A0_prime, decay_const, decay_const_prime, t_irr-dt_irr/2))
    dfdx_list.append(dfdt_irr)
    unc_list.append(t_irr_unc)

    dfdx = np.array(dfdx_list)
    unc = np.array(unc_list)
    ratio_unc = np.sqrt(np.sum(np.multiply(dfdx,dfdx)* np.multiply(unc,unc)))

    return ratio, ratio_unc



def foil_energy_w_unc(E_array, iaea_ratio, exp_ratio, exp_ratio_unc): 
#E_array and iaea_ratio are arrays, while exp_ratio and exp_ratio_unc are numbers
    
    #Finding the energy:______________________________________________________________
    exp_ratio_array = np.zeros(len(iaea_ratio))
    exp_ratio_array.fill(exp_ratio)
    dE = E_array[1]-E_array[0]
    numb_of_indices_5MeV = np.abs(5/dE)
    # dratio = 1e-5
    indices = np.nonzero(np.abs(iaea_ratio - exp_ratio_array)<exp_ratio_unc) #indices of all the possible enrgies
    possible_energies_w_duplicates = E_array[indices] #array with all the possible enrgies
    iaea_ratio_for_possible_energies = iaea_ratio[indices] #array with corresponding iaea ratio for all the possible enrgies
    # print('Energies with duplicates: ', E_array[indices])

    minimumEnergyDifferenceForNewSolution = 5 #[MeV] #This can be sett to 0.5 for the 25MeV stack and 5 for the 55MeV stack
    ratioDifference = [np.abs(iaea_ratio_for_possible_energies[0]-exp_ratio)]
    ratio_solutions = []
    energy_solutions = []

    #Storing the first set of numbers to compare:
    ratio_solutions.append(iaea_ratio_for_possible_energies[0])
    energy_solutions.append(possible_energies_w_duplicates[0])

    #Finding all the energy solutions:
    for i in range(len(iaea_ratio_for_possible_energies)): 
        #i loops through all possible energies
        ratio_diff = np.abs(iaea_ratio_for_possible_energies[i]-exp_ratio)
        energy_diff = np.abs(possible_energies_w_duplicates[i]-energy_solutions[-1])

        newRegion = False
        if (energy_diff>minimumEnergyDifferenceForNewSolution): #check if we have a new possible energy (ratio line crosses multiple times?)
            newRegion = True 
            
        if (newRegion):
            ratioDifference.append(ratio_diff)
            ratio_solutions.append(iaea_ratio_for_possible_energies[i])
            energy_solutions.append(possible_energies_w_duplicates[i])

        else:
            if (ratio_diff<ratioDifference[-1]):
                ratioDifference[-1]=ratio_diff
                ratio_solutions[-1]=iaea_ratio_for_possible_energies[i]
                energy_solutions[-1]=possible_energies_w_duplicates[i]

    solution_indices = []
    for energy in energy_solutions:
        index = np.argmin(np.abs(E_array-energy))
        solution_indices.append(index)

    energy_unc_minus = []
    energy_unc_plus = []

    #Now we need to find the uncertainty for each energy solution:
    for index in solution_indices:
        index_m_5 = int(index-numb_of_indices_5MeV)
        index_p_5 = int(index+numb_of_indices_5MeV)
        if (index_m_5<0):
            index_m_5 = 0
        if (index_p_5>len(iaea_ratio)):
            index_p_5 = -1

        if (iaea_ratio[index]<iaea_ratio[index+1]):
            if (iaea_ratio[index_m_5]>iaea_ratio[index]): #Hvis bunnpunkt
                index_m = np.argmin(np.abs(iaea_ratio[index_m_5:index]-(exp_ratio+exp_ratio_unc)))
                index_p = np.argmin(np.abs(iaea_ratio[index:index_p_5]-(exp_ratio+exp_ratio_unc)))
                index_minus = np.argmin(np.abs(iaea_ratio-iaea_ratio[index_m_5:index][index_m]))
                index_plus = np.argmin(np.abs(iaea_ratio-iaea_ratio[index:index_p_5][index_p]))

                energy_unc_minus.append(np.abs(E_array[index]-E_array[index_minus]))
                energy_unc_plus.append(np.abs(E_array[index_plus]-E_array[index])) 

            else:    
                index_m = np.argmin(np.abs(iaea_ratio[index_m_5:index]-(exp_ratio-exp_ratio_unc)))
                index_p = np.argmin(np.abs(iaea_ratio[index:index_p_5]-(exp_ratio+exp_ratio_unc)))
                index_minus = np.argmin(np.abs(iaea_ratio-iaea_ratio[index_m_5:index][index_m]))
                index_plus = np.argmin(np.abs(iaea_ratio-iaea_ratio[index:index_p_5][index_p]))

                energy_unc_minus.append(np.abs(E_array[index]-E_array[index_minus]))
                energy_unc_plus.append(np.abs(E_array[index_plus]-E_array[index]))            

        elif (iaea_ratio[index]>iaea_ratio[index+1]):
            
            index_m = np.argmin(np.abs(iaea_ratio[index_m_5:index]-(exp_ratio+exp_ratio_unc)))
            index_p = np.argmin(np.abs(iaea_ratio[index:index_p_5]-(exp_ratio-exp_ratio_unc)))
            index_minus = np.argmin(np.abs(iaea_ratio-iaea_ratio[index_m_5:index][index_m]))
            index_plus = np.argmin(np.abs(iaea_ratio-iaea_ratio[index:index_p_5][index_p]))

            energy_unc_minus.append(np.abs(E_array[index]-E_array[index_minus]))
            energy_unc_plus.append(np.abs(E_array[index_plus]-E_array[index])) 

        else:
            print('Something about the energy uncertainty does not make sence...')

    return energy_solutions, energy_unc_plus, energy_unc_minus




def ratios_and_energies(A0, t_half, t_irr, A0_unc, t_half_unc, t_irr_unc, E_array, ratios_iaea):

    ratios = []
    ratios_unc = []
    E_foils = []
    E_foils_unc_plus = []
    E_foils_unc_minus = []

    for i in range(0,len(A0), 2):
        ratio, ratio_unc = exp_xs_ratio_from_A0_w_unc(A0[i], A0[i+1], t_half[0], t_half[1], t_irr, A0_unc[i], A0_unc[i+1], t_half_unc[0], t_half_unc[1], t_irr_unc)
        E_foil, E_foil_unc_plus, E_foil_unc_minus = foil_energy_w_unc(E_array, ratios_iaea, ratio, ratio_unc)

        ratios.append(ratio)
        ratios_unc.append(ratio_unc)
        E_foils.append(E_foil)
        E_foils_unc_plus.append(E_foil_unc_plus)
        E_foils_unc_minus.append(E_foil_unc_minus)

    return ratios, ratios_unc, E_foils, E_foils_unc_plus, E_foils_unc_minus 

  


def print_and_plot_energies_and_ratios(ratios_iaea_Ti, ratios_iaea_Cu, E_array, E_Ti_foils_paper, E_Ti_foils_unc_paper, E_Cu_foils_paper, E_Cu_foils_unc_paper, ratios_Ti, ratios_unc_Ti, E_Ti_foils, E_Ti_foils_unc_plus, E_Ti_foils_unc_minus, ratios_Cu, ratios_unc_Cu, E_Cu_foils, E_Cu_foils_unc_plus, E_Cu_foils_unc_minus, colors, stack_name):

    #Printing the energies
    for i in range(len(E_Ti_foils)):
        print(f'Energy in Ti foil number {i+1} form the paper is:')
        print(f'{E_Ti_foils_paper[i]:.2f} +- {E_Ti_foils_unc_paper[i]:.2f} MeV')

        print(f'Energy solutions in Ti foil number {i+1}:')
        for j in range(len(E_Ti_foils[i])):
            print(f'{E_Ti_foils[i][j]:.2f} + {E_Ti_foils_unc_plus[i][j]:.2f} - {E_Ti_foils_unc_minus[i][j]:.2f} MeV')
        print(' ')

    for i in range(len(E_Cu_foils)):
        print(f'Energy in Cu foil number {i+1} form the paper is:')
        print(f'{E_Cu_foils_paper[i]:.2f} +- {E_Cu_foils_unc_paper[i]:.2f} MeV ')

        print(f'Energy solutions in Cu foil number {i+1}:')
        for j in range(len(E_Cu_foils[i])):
            print(f'{E_Cu_foils[i][j]:.2f} + {E_Cu_foils_unc_plus[i][j]:.2f} - {E_Cu_foils_unc_minus[i][j]:.2f} MeV')
        print(' ')


    #Plot energies and ratios:
    #Starting with the Ti foils:
    plt.subplot(121)
    #plotter IAEA ratio:
    plt.plot(E_array, ratios_iaea_Ti , label = 'IAEA', color = 'black')
    #plotter exp ratio and energy with unc:
    for i in range(len(E_Ti_foils)):
        #Ratio:
        plt.plot([E_array[0],E_array[-1]], [ratios_Ti[i], ratios_Ti[i]], label = f'Foil {i+1}', color = colors[i])
        plt.plot([E_array[0],E_array[-1]], [ratios_Ti[i]+ratios_unc_Ti[i], ratios_Ti[i]+ratios_unc_Ti[i]], color = colors[i], linestyle = 'dashed')
        plt.plot([E_array[0],E_array[-1]], [ratios_Ti[i]-ratios_unc_Ti[i], ratios_Ti[i]-ratios_unc_Ti[i]], color = colors[i], linestyle = 'dashed')
        for j in range(len(E_Ti_foils[i])):
            #Energy:
            plt.plot([E_Ti_foils[i][j], E_Ti_foils[i][j]], [0,4], color = colors[i])
            plt.plot([E_Ti_foils[i][j]+E_Ti_foils_unc_plus[i][j], E_Ti_foils[i][j]+E_Ti_foils_unc_plus[i][j]], [0,4], color = colors[i], linestyle = 'dashed')
            plt.plot([E_Ti_foils[i][j]-E_Ti_foils_unc_minus[i][j], E_Ti_foils[i][j]-E_Ti_foils_unc_minus[i][j]], [0,4], color = colors[i], linestyle = 'dashed')
    plt.title(f'natTi(p,x)46Sc/natTi(p,x)48V in the {stack_name} stack')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.legend()

    #Cu foils:
    plt.subplot(122)
    #plotter IAEA ratio:
    plt.plot(E_array, ratios_iaea_Cu , label = 'IAEA', color = 'black')
    #plotter exp ratio and energy with unc:
    for i in range(len(E_Cu_foils)):
        #Ratio:
        plt.plot([E_array[0],E_array[-1]], [ratios_Cu[i], ratios_Cu[i]], label = f'Foil {i+1}', color = colors[i])
        plt.plot([E_array[0],E_array[-1]], [ratios_Cu[i]+ratios_unc_Cu[i], ratios_Cu[i]+ratios_unc_Cu[i]], color = colors[i], linestyle = 'dashed')
        plt.plot([E_array[0],E_array[-1]], [ratios_Cu[i]-ratios_unc_Cu[i], ratios_Cu[i]-ratios_unc_Cu[i]], color = colors[i], linestyle = 'dashed')
        for j in range(len(E_Cu_foils[i])):
            #Energy:
            plt.plot([E_Cu_foils[i][j], E_Cu_foils[i][j]], [0,4], color = colors[i])
            plt.plot([E_Cu_foils[i][j]+E_Cu_foils_unc_plus[i][j], E_Cu_foils[i][j]+E_Cu_foils_unc_plus[i][j]], [0,4], color = colors[i], linestyle = 'dashed')
            plt.plot([E_Cu_foils[i][j]-E_Cu_foils_unc_minus[i][j], E_Cu_foils[i][j]-E_Cu_foils_unc_minus[i][j]], [0,4], color = colors[i], linestyle = 'dashed')
    plt.title(f'natCu(p,x)62Zn/natCu(p,x)63Zn in the {stack_name} stack')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.legend()
    plt.show()







if __name__=='__main__': #______________________________________________________________________________________________________

    colors = ['hotpink', 'lightcoral', 'orange', 'gold', 'limegreen', 'lightseagreen', 'deepskyblue', 'mediumslateblue', 'orchid']
    
    #The same for both stacks:
    reaction_list_voyles = ['natTi_46Sc', 'natTi_48V', 'natCu_62Zn', 'natCu_63Zn']
    #Ti half lifes:
    t_half_Ti = np.array([83.79*60*60*24, 15.9735*60*60*24]) #[s]
    t_half_unc_Ti = np.array([0.04*60*60*24, 0.0025*60*60*24]) #[s]
    #Cu half lives:
    t_half_Cu = np.array([9.186*60*60, 38.47*60]) #[s]
    t_half_unc_Cu = np.array([0.013*60*60, 0.05*60]) #[s]



    # Voyles 25MeV stack info: ______________________________________________________________________________________________________________________________________________________
    t_irr_25MeV = 20*60 #[s]
    t_irr_unc_25MeV = 5 #[s]
    beam_voyles = 'p'

    #Ti lists
    A0_Ti_25MeV = [70.5041, 3080.23, 49.9634, 5085.61] #[Bq]
    A0_unc_Ti_25MeV = [1.6824, 1.99114, 1.031, 119.1] #[Bq]

    #Cu lists
    A0_Cu_25MeV = [161744, 698847, 116562, 1.90e+006, 34046.1, 5.48e+006] #[Bq]
    A0_unc_Cu_25MeV = [1805, 2.30e+004, 1994, 5.74e+004, 433.1, 1.38e+005] #[Bq]

    #Lists of energies and unc from the paper:
    E_Ti_foils_paper_25MeV = [22.29, 18.98] #[MeV]
    E_Ti_foils_unc_paper_25MeV = [0.32, 0.37] #[MeV]
    E_Cu_foils_paper_25MeV = [21.70, 18.30, 15.38] #[MeV]
    E_Cu_foils_unc_paper_25MeV = [0.33, 0.38, 0.44] #[MeV]


    # Voyles 55MeV stack info: ______________________________________________________________________________________________________________________________________________________
    t_irr_55MeV = 10*60 #[s]
    t_irr_unc_55MeV = 5 #[s]

    #Ti lists
    A0_Ti_55MeV = [334.88, 486.933, 371.493, 586.622, 315.986, 769.791, 191.424, 908.55, 73.5383, 1020.04, 40.3647, 1459.17] #[Bq]
    A0_unc_Ti_55MeV = [4.68, 14.12, 2.191, 10.63, 2.189, 37.06, 1.9454, 33.95, 1.9454, 33.95, 1.9869, 15.2, 1.228, 65.76] #[Bq]


    #Cu lists
    A0_Cu_55MeV = [19881.3, 373546, 21026.4, 536831, 22856, 1.02E+006, 33332.4, 1.24E+006, 73083.4, 9.35E+005, 131635, 552302, 45045.2, 3.05E+006] #[Bq]
    A0_unc_Cu_55MeV = [409.1, 1.70E+004, 333.5, 1.10E+004, 561.8, 3.72E+004, 966.5, 4.37E+004, 1903, 3.39E+004, 2135, 1.91E+004, 905.3, 1.21E+005] #[Bq]
    A0_Cu_55MeV = [19881.3, 373546, 21026.4, 536831, 22856, 1.02E+006, 33332.4, 1.24E+006, 73083.4, 9.35E+005,     45045.2, 3.05E+006] #Have removed foil 6 
    A0_unc_Cu_55MeV = [409.1, 1.70E+004, 333.5, 1.10E+004, 561.8, 3.72E+004, 966.5, 4.37E+004, 1903, 3.39E+004,     905.3, 1.21E+005]      

    #Lists of energies and unc from the paper:
    E_Ti_foils_paper_55MeV = [53.31, 46.48, 38.76, 34.44, 29.63, 24.1] #[MeV]
    E_Ti_foils_unc_paper_55MeV = [0.61, 0.68, 0.78, 0.86, 0.96, 1.1] #[MeV]
    E_Cu_foils_paper_55MeV = [53.04, 46.18, 38.42, 34.06, 29.21, 23.6, 16.6] #[MeV]
    E_Cu_foils_unc_paper_55MeV = [0.61, 0.68, 0.79, 0.86, 0.97, 1.2, 1.5] #[MeV]
    E_Cu_foils_paper_55MeV = [53.04, 46.18, 38.42, 34.06, 29.21, 16.6] #[MeV]
    E_Cu_foils_unc_paper_55MeV = [0.61, 0.68, 0.79, 0.86, 0.97, 1.5] #[MeV]


    #IAEA ratios for both Voyles stacks:
    E_iaea_protons, xs_iaea_protons, xs_unc_iaea_protons = generate_xs_lists(beam_voyles, reaction_list_voyles)
    E_voyles = np.linspace(9,60, 1000000)
    xs_interpol_iaea_Ti_46Sc_protons = interpol_xs_iaea(xs_iaea_protons[0], E_iaea_protons[0])
    xs_interpol_iaea_Ti_48V_protons = interpol_xs_iaea(xs_iaea_protons[1], E_iaea_protons[1])
    xs_interpol_iaea_Cu_62Zn_protons = interpol_xs_iaea(xs_iaea_protons[2], E_iaea_protons[2])
    xs_interpol_iaea_Cu_63Zn_protons = interpol_xs_iaea(xs_iaea_protons[3], E_iaea_protons[3])
    ratios_iaea_Ti_protons = xs_interpol_iaea_Ti_46Sc_protons(E_voyles)/xs_interpol_iaea_Ti_48V_protons(E_voyles)
    ratios_iaea_Cu_protons = xs_interpol_iaea_Cu_62Zn_protons(E_voyles)/xs_interpol_iaea_Cu_63Zn_protons(E_voyles)



    # Fox LBNL stack with A0: ________________________________________________________________________________________________________________________________________________________
   
    reaction_list_fox = ['natTi_46Sc', 'natTi_48V', 'natCu_62Zn', 'natCu_63Zn']
    t_irr_fox = 3884 #[s]
    t_irr_unc_fox = 5 #[s]
    beam_fox = 'p'

    #Ti lists
    A0_Ti_fox = [3496.71, 4807.33, 3614.95, 5701.48, 3906.91, 5790.39, 4180.33, 6496.83, 3960.36, 6808.39, 4054.90, 6656.59, 1502.79, 2935.30, 3045.40, 8675.28, 2537.80, 
                 9361.61] #[Bq]
    A0_unc_Ti_fox = [40.56, 103.94, 47.38, 115.98, 49.07, 134.40, 157.20, 280.51, 100.50, 345.38, 90.77, 189.19, 22.56, 85.81, 80.38, 247.57, 35.90, 154.23] #[Bq]

    #Cu lists
    A0_Cu_fox = [227784.39, 3058839.03, 231124.48, 3520974.67, 252054.55, 3409080.13, 250212.80, 4348070.52, 256412.89, 4605124.88, 245163.28, 4943754.57, 255262.49, 
                 6296766.00, 272322.45, 7223439.48, 317728.16, 9551722.20] #[Bq]
    A0_unc_Cu_fox = [7841.01, 257126.71, 8029.79, 412942.12, 8620.42, 300512.28, 8569.40, 377332.86, 8815.24, 864277.78, 8394.78, 717498.24, 8766.95, 607172.74, 9527.21,
                     590380.39, 10919.11, 861212.48] #[Bq]

    #Lists of energies and unc from the paper:
    E_Ti_foils_paper_fox = [54.9, 51.9, 49.5, 46.9, 45.4, 43.5, 41.9, 38.0, 36.2] #[MeV]
    E_Ti_foils_unc_paper_fox = [1.3, 1.4, 1.4, 1.5, 1.5, 1.6, 1.6, 1.7, 1.8] #[MeV]
    E_Cu_foils_paper_fox = [55.2, 52.2, 49.9, 47.3, 45.8, 43.9, 42.3, 38.4, 36.7] #[MeV]
    E_Cu_foils_unc_paper_fox = [1.3, 1.4, 1.4, 1.5, 1.5, 1.6, 1.6, 1.7, 1.8] #[MeV]


    #IAEA ratios for Fox stack:
    E_iaea_protons, xs_iaea_protons, xs_unc_iaea_protons = generate_xs_lists(beam_voyles, reaction_list_fox)
    E_fox = np.linspace(9,60, 1000000)
    xs_interpol_iaea_Ti_46Sc_protons = interpol_xs_iaea(xs_iaea_protons[0], E_iaea_protons[0])
    xs_interpol_iaea_Ti_48V_protons = interpol_xs_iaea(xs_iaea_protons[1], E_iaea_protons[1])
    xs_interpol_iaea_Cu_62Zn_protons = interpol_xs_iaea(xs_iaea_protons[2], E_iaea_protons[2])
    xs_interpol_iaea_Cu_63Zn_protons = interpol_xs_iaea(xs_iaea_protons[3], E_iaea_protons[3])
    ratios_iaea_Ti_protons = xs_interpol_iaea_Ti_46Sc_protons(E_fox)/xs_interpol_iaea_Ti_48V_protons(E_fox)
    ratios_iaea_Cu_protons = xs_interpol_iaea_Cu_62Zn_protons(E_fox)/xs_interpol_iaea_Cu_63Zn_protons(E_fox)





    #Running functions: ______________________________________________________________________________________________________________________________________________________
    #Running the functions for the Voyles 25MeV stack:
    print('_______________________25 MeV stack_______________________')
    ratios_Ti_25MeV, ratios_unc_Ti_25MeV, E_Ti_foils_25MeV, E_Ti_foils_unc_plus_25MeV, E_Ti_foils_unc_minus_25MeV = ratios_and_energies(A0_Ti_25MeV, t_half_Ti, t_irr_25MeV, A0_unc_Ti_25MeV, t_half_unc_Ti, t_irr_unc_25MeV, E_voyles, ratios_iaea_Ti_protons)

    ratios_Cu_25MeV, ratios_unc_Cu_25MeV, E_Cu_foils_25MeV, E_Cu_foils_unc_plus_25MeV, E_Cu_foils_unc_minus_25MeV = ratios_and_energies(A0_Cu_25MeV, t_half_Cu, t_irr_25MeV, A0_unc_Cu_25MeV, t_half_unc_Cu, t_irr_unc_25MeV, E_voyles, ratios_iaea_Cu_protons)

    print_and_plot_energies_and_ratios(ratios_iaea_Ti_protons, ratios_iaea_Cu_protons, E_voyles, E_Ti_foils_paper_25MeV, E_Ti_foils_unc_paper_25MeV, E_Cu_foils_paper_25MeV, E_Cu_foils_unc_paper_25MeV, ratios_Ti_25MeV, ratios_unc_Ti_25MeV, E_Ti_foils_25MeV, E_Ti_foils_unc_plus_25MeV, E_Ti_foils_unc_minus_25MeV, ratios_Cu_25MeV, ratios_unc_Cu_25MeV, E_Cu_foils_25MeV, E_Cu_foils_unc_plus_25MeV, E_Cu_foils_unc_minus_25MeV, colors, 'Voyles 25MeV')



    #Running the functions for the Voyles 55MeV stack:
    print('_______________________55 MeV stack_______________________')
    ratios_Ti_55MeV, ratios_unc_Ti_55MeV, E_Ti_foils_55MeV, E_Ti_foils_unc_plus_55MeV, E_Ti_foils_unc_minus_55MeV = ratios_and_energies(A0_Ti_55MeV, t_half_Ti, t_irr_55MeV, A0_unc_Ti_55MeV, t_half_unc_Ti, t_irr_unc_55MeV, E_voyles, ratios_iaea_Ti_protons)

    ratios_Cu_55MeV, ratios_unc_Cu_55MeV, E_Cu_foils_55MeV, E_Cu_foils_unc_plus_55MeV, E_Cu_foils_unc_minus_55MeV = ratios_and_energies(A0_Cu_55MeV, t_half_Cu, t_irr_55MeV, A0_unc_Cu_55MeV, t_half_unc_Cu, t_irr_unc_55MeV, E_voyles, ratios_iaea_Cu_protons)

    print_and_plot_energies_and_ratios(ratios_iaea_Ti_protons, ratios_iaea_Cu_protons, E_voyles, E_Ti_foils_paper_55MeV, E_Ti_foils_unc_paper_55MeV, E_Cu_foils_paper_55MeV, E_Cu_foils_unc_paper_55MeV, ratios_Ti_55MeV, ratios_unc_Ti_55MeV, E_Ti_foils_55MeV, E_Ti_foils_unc_plus_55MeV, E_Ti_foils_unc_minus_55MeV, ratios_Cu_55MeV, ratios_unc_Cu_55MeV, E_Cu_foils_55MeV, E_Cu_foils_unc_plus_55MeV, E_Cu_foils_unc_minus_55MeV, colors, 'Voyles 55MeV')



    #Running the functions for the Fox LBNL stack:
    print('_______________________Fox LBNL stack_______________________')
    ratios_Ti_fox, ratios_unc_Ti_fox, E_Ti_foils_fox, E_Ti_foils_unc_plus_fox, E_Ti_foils_unc_minus_fox = ratios_and_energies(A0_Ti_fox, t_half_Ti, t_irr_fox, A0_unc_Ti_fox, t_half_unc_Ti, t_irr_unc_fox, E_fox, ratios_iaea_Ti_protons)

    ratios_Cu_fox, ratios_unc_Cu_fox, E_Cu_foils_fox, E_Cu_foils_unc_plus_fox, E_Cu_foils_unc_minus_fox = ratios_and_energies(A0_Cu_fox, t_half_Cu, t_irr_fox, A0_unc_Cu_fox, t_half_unc_Cu, t_irr_unc_fox, E_fox, ratios_iaea_Cu_protons)

    print_and_plot_energies_and_ratios(ratios_iaea_Ti_protons, ratios_iaea_Cu_protons, E_fox, E_Ti_foils_paper_fox, E_Ti_foils_unc_paper_fox, E_Cu_foils_paper_fox, E_Cu_foils_unc_paper_fox, ratios_Ti_fox, ratios_unc_Ti_fox, E_Ti_foils_fox, E_Ti_foils_unc_plus_fox, E_Ti_foils_unc_minus_fox, ratios_Cu_fox, ratios_unc_Cu_fox, E_Cu_foils_fox, E_Cu_foils_unc_plus_fox, E_Cu_foils_unc_minus_fox, colors, 'Fox LBNL')






