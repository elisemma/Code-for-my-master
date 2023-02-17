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
    dratio = 1e-5
    indices = np.nonzero(np.abs(iaea_ratio - exp_ratio_array)<dratio) #indices of all the possible enrgies
    possible_energies_w_duplicates = E_array[indices] #array with all the possible enrgies
    iaea_ratio_for_possible_energies = iaea_ratio[indices] #array with corresponding iaea ratio for all the possible enrgies
    # print('Energies with duplicates: ', E_array[indices])

    minimumEnergyDifferenceForNewSolution = 0.1 #[MeV]
    ratioDifference = [np.abs(iaea_ratio_for_possible_energies[0]-exp_ratio)]
    ratio_solutions = []
    energy_solutions = []

    #Storing the first set of numbers to compare:
    ratio_solutions.append(iaea_ratio_for_possible_energies[0])
    energy_solutions.append(possible_energies_w_duplicates[0])

    #Finding all the energy solutions:
    for i in range(len(iaea_ratio_for_possible_energies)): 
        #i loops through all possible energies
        for j in range(len(ratio_solutions)):
            # j loop through all solutions you have stored (different solutions being defined to be having 
            #a difference in energy of at least "minimumEnergyDifferenceForNewSolution")
            ratio_diff = np.abs(iaea_ratio_for_possible_energies[i]-exp_ratio)
            energy_diff = np.abs(possible_energies_w_duplicates[i]-energy_solutions[j])

            newRegion = False
            if (energy_diff>minimumEnergyDifferenceForNewSolution): #check if we have a new possible energy (ratio line crosses multiple times?)
                newRegion = True 

            if (newRegion):
                ratioDifference.append(ratio_diff)
                ratio_solutions.append(iaea_ratio_for_possible_energies[i])
                energy_solutions.append(possible_energies_w_duplicates[i])

            else:
                if (ratio_diff<ratioDifference[j]):
                    ratioDifference[j]=ratio_diff
                    ratio_solutions[j]=iaea_ratio_for_possible_energies[i]
                    energy_solutions[j]=possible_energies_w_duplicates[i]


    solution_indices = []
    for energy in energy_solutions:
        index = np.argmin(np.abs(E_array-energy))
        solution_indices.append(index)

    energy_unc_minus = []
    energy_unc_plus = []

    #Now we need to find the uncertainty for each energy solution:
    for index in solution_indices:
        #starter med å lete etter den nedre grensa for energien help
        if (iaea_ratio[index]<iaea_ratio[index+1]):
            #må bruke index_minus 
            index_m = np.argmin(np.abs(iaea_ratio[0:index]-(exp_ratio-exp_ratio_unc)))
            index_p = np.argmin(np.abs(iaea_ratio[index:-1]-(exp_ratio+exp_ratio_unc)))
            index_minus = np.argmin(np.abs(iaea_ratio-iaea_ratio[0:index][index_m]))
            index_plus = np.argmin(np.abs(iaea_ratio-iaea_ratio[index:-1][index_p]))


            energy_unc_minus.append(np.abs(E_array[index]-E_array[index_minus]))
            energy_unc_plus.append(np.abs(E_array[index_plus]-E_array[index]))            

        elif (iaea_ratio[index]>iaea_ratio[index+1]):
            #må bruke øvre stipla linje her
            index_m = np.argmin(np.abs(iaea_ratio[0:index]-(exp_ratio+exp_ratio_unc)))
            index_p = np.argmin(np.abs(iaea_ratio[index:-1]-(exp_ratio-exp_ratio_unc)))
            index_minus = np.argmin(np.abs(iaea_ratio-iaea_ratio[0:index][index_m]))
            index_plus = np.argmin(np.abs(iaea_ratio-iaea_ratio[index:-1][index_p]))

            energy_unc_minus.append(np.abs(E_array[index]-E_array[index_minus]))
            energy_unc_plus.append(np.abs(E_array[index_plus]-E_array[index])) 

        else:
            print('Something about the energy uncertainty does not make sence...')

    


    # #Finding the uncertainty:_________________________________________________________
    # exp_ratio_unc_array = np.zeros(len(iaea_ratio))
    # exp_ratio_unc_array.fill(exp_ratio_unc)
    # #Making new energy arrays with energies from E_foil-5MeV up to E_foil+5MeV:
    # index_m_5MeV = np.argmin(np.abs(E_array - (E_foil - 5)))
    # index_p_5MeV = np.argmin(np.abs(E_array - (E_foil + 5)))

    # E_foil_m_5MeV = E_array[index_m_5MeV] #find E_foil - 5MeV
    # E_foil_p_5MeV = E_array[index_p_5MeV] #find E_foil + 5MeV

    # E_p_5MeV_array = np.where(E_array<E_foil_p_5MeV, E_array, 0) #energy array where all elements bigger than E_foil+5MeV is set to zero
    # E_pm_5MeV_array_w_zeros = np.where(E_p_5MeV_array>E_foil_m_5MeV, E_array, 0) #energy array where all elements smaller than E_foil-5MeV is set to zero
    # E_pm_5MeV_index = np.nonzero(E_pm_5MeV_array_w_zeros>0) #find the indecies of the nonzero elements in the energy array 

    # E_pm_5MeV_array = E_array[E_pm_5MeV_index] #new energy array which only includes energies from foil energy minus 5MeV to foil energy plus 5MeV
    # iaea_ratio_pm_5MeV = iaea_ratio[E_pm_5MeV_index] #find the corresponding ratios


    #Finding the indices of the uncertainties 
    
    # index_plus = np.nonzero(np.abs(iaea_ratio-(exp_ratio_array+exp_ratio_unc_array))<dratio)
    # index_minus = np.nonzero(np.abs(iaea_ratio-(exp_ratio_array-exp_ratio_unc_array))<dratio)

    # print('iaea_ratio[index_plus]: ', iaea_ratio[index_plus])
    # print('iaea_ratio[index_minus]: ', iaea_ratio[index_minus])


    return energy_solutions, energy_unc_plus, energy_unc_minus








if __name__=='__main__': #______________________________________________________________________________________________________

    reaction_list = ['natTi_46Sc', 'natTi_48V', 'natCu_62Zn', 'natCu_63Zn']
    A0 = [70.5041, 3080.23, 161744, 698847] #[Bq]
    A0_unc = [1.6824, 1.99114, 1805, 2.30e+004] #[Bq]
    t_half = np.array([83.79*60*60*24, 15.9735*60*60*24, 9.186*60*60, 38.47*60]) #[s]
    t_half_unc = np.array([0.04*60*60*24, 0.0025*60*60*24, 0.013*60*60, 0.05*60]) #[s]
    decay_const = np.log(2)/t_half #[1/s]
    t_irr = 20*60 #[s]
    t_irr_unc = 5 #[s]
    beam = 'p'

    ratio_Ti_46Sc_div_Ti_48V_exsp, ratio_Ti_46Sc_div_Ti_48V_exsp_unc = exp_xs_ratio_from_A0_w_unc(A0[0], A0[1], t_half[0], t_half[1], t_irr, A0_unc[0], A0_unc[1], t_half_unc[0], t_half_unc[1], t_irr_unc)

    ratio_Cu_62Zn_div_Cu_63Zn_exsp, ratio_Cu_62Zn_div_Cu_63Zn_exsp_unc = exp_xs_ratio_from_A0_w_unc(A0[2], A0[3], t_half[2], t_half[3], t_irr, A0_unc[2], A0_unc[3], t_half_unc[2], t_half_unc[3], t_irr_unc)

    print(f'exsp rato for 48V: {ratio_Ti_46Sc_div_Ti_48V_exsp} pm {ratio_Ti_46Sc_div_Ti_48V_exsp_unc}')
    print(f'exsp rato for 62Zn: {ratio_Cu_62Zn_div_Cu_63Zn_exsp} pm {ratio_Cu_62Zn_div_Cu_63Zn_exsp_unc}')



    E_iaea, xs_iaea, xs_unc_iaea = generate_xs_lists(beam, reaction_list)

    E = np.linspace(10,60, 1000000)
    xs_interpol_iaea_Ti_46Sc = interpol_xs_iaea(xs_iaea[0], E_iaea[0])
    xs_interpol_iaea_Ti_48V = interpol_xs_iaea(xs_iaea[1], E_iaea[1])
    xs_interpol_iaea_Cu_62Zn = interpol_xs_iaea(xs_iaea[2], E_iaea[2])
    xs_interpol_iaea_Cu_63Zn = interpol_xs_iaea(xs_iaea[3], E_iaea[3])

    ratio_Ti_46Sc_div_Ti_48V_iaea = xs_interpol_iaea_Ti_46Sc(E)/xs_interpol_iaea_Ti_48V(E)
    ratio_Cu_62Zn_div_Cu_63Zn_iaea = xs_interpol_iaea_Cu_62Zn(E)/xs_interpol_iaea_Cu_63Zn(E)


    E_Ti_foil, E_Ti_foil_unc_plus, E_Ti_foil_unc_minus = foil_energy_w_unc(E, ratio_Ti_46Sc_div_Ti_48V_iaea, ratio_Ti_46Sc_div_Ti_48V_exsp, ratio_Ti_46Sc_div_Ti_48V_exsp_unc)
    E_Cu_foil, E_Cu_foil_unc_plus, E_Cu_foil_unc_minus = foil_energy_w_unc(E, ratio_Cu_62Zn_div_Cu_63Zn_iaea, ratio_Cu_62Zn_div_Cu_63Zn_exsp, ratio_Cu_62Zn_div_Cu_63Zn_exsp_unc)


    print(f'The energy in the first Ti-foil is {E_Ti_foil[0]:.2f} + {E_Ti_foil_unc_plus[0]:.2f} - {E_Ti_foil_unc_minus[0]:.2f} MeV')
    print(f'The energy in the first Cu-foil is {E_Cu_foil[0]:.2f} + {E_Cu_foil_unc_plus[0]:.2f} - {E_Cu_foil_unc_minus[0]:.2f} MeV')
    print(f'                                or {E_Cu_foil[1]:.2f} + {E_Cu_foil_unc_plus[1]:.2f} - {E_Cu_foil_unc_minus[1]:.2f} MeV')


    plt.subplot(121)
    #plotter IAEA ratio:
    plt.plot(E, ratio_Ti_46Sc_div_Ti_48V_iaea , label = 'IAEA: natTi(p,x)46Sc/natTi(p,x)48V', color = 'hotpink')
    #plotter exp ratio with unc:
    plt.plot([E[0],E[-1]], [ratio_Ti_46Sc_div_Ti_48V_exsp, ratio_Ti_46Sc_div_Ti_48V_exsp], label = 'Voyles2021: natTi(p,x)46Sc/natTi(p,x)48V', color = 'pink')
    plt.plot([E[0],E[-1]], [ratio_Ti_46Sc_div_Ti_48V_exsp+ratio_Ti_46Sc_div_Ti_48V_exsp_unc, ratio_Ti_46Sc_div_Ti_48V_exsp+ratio_Ti_46Sc_div_Ti_48V_exsp_unc], color = 'violet', linestyle = 'dashed')
    plt.plot([E[0],E[-1]], [ratio_Ti_46Sc_div_Ti_48V_exsp-ratio_Ti_46Sc_div_Ti_48V_exsp_unc, ratio_Ti_46Sc_div_Ti_48V_exsp-ratio_Ti_46Sc_div_Ti_48V_exsp_unc], color = 'm', linestyle = 'dashed')
    #plotter energy with unc:
    plt.plot([E_Ti_foil[0], E_Ti_foil[0]], [0,4], label = 'Voyles2021: natTi(p,x)46Sc/natTi(p,x)48V', color = 'pink')
    plt.plot([E_Ti_foil[0]+E_Ti_foil_unc_plus[0], E_Ti_foil[0]+E_Ti_foil_unc_plus[0]], [0,4], color = 'violet', linestyle = 'dashed')
    plt.plot([E_Ti_foil[0]-E_Ti_foil_unc_minus[0], E_Ti_foil[0]-E_Ti_foil_unc_minus[0]], [0,4], color = 'm', linestyle = 'dashed')

    plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.title('First compartment in the 25 MeV stack')
    plt.legend()


    plt.subplot(122)
    #plotter IAEA ratio:
    plt.plot(E, ratio_Cu_62Zn_div_Cu_63Zn_iaea , label = 'IAEA: natCu(p,x)62Zn/natCu(p,x)63Zn', color = 'deepskyblue')
    #plotter exp ratio with unc:
    plt.plot([E[0],E[-1]], [ratio_Cu_62Zn_div_Cu_63Zn_exsp, ratio_Cu_62Zn_div_Cu_63Zn_exsp], label = 'Voyles2021: natCu(p,x)62Zn/natCu(p,x)63Zn', color = 'lightskyblue')
    plt.plot([E[0],E[-1]], [ratio_Cu_62Zn_div_Cu_63Zn_exsp+ratio_Cu_62Zn_div_Cu_63Zn_exsp_unc, ratio_Cu_62Zn_div_Cu_63Zn_exsp+ratio_Cu_62Zn_div_Cu_63Zn_exsp_unc], label = 'Voyles2021: natCu(p,x)62Zn/natCu(p,x)63Zn', color = 'lightskyblue', linestyle = 'dashed')
    plt.plot([E[0],E[-1]], [ratio_Cu_62Zn_div_Cu_63Zn_exsp-ratio_Cu_62Zn_div_Cu_63Zn_exsp_unc, ratio_Cu_62Zn_div_Cu_63Zn_exsp-ratio_Cu_62Zn_div_Cu_63Zn_exsp_unc], label = 'Voyles2021: natCu(p,x)62Zn/natCu(p,x)63Zn', color = 'lightskyblue', linestyle = 'dashed')
    #plotter energy with unc:
    plt.plot([E_Cu_foil[0], E_Cu_foil[0]], [0,4], label = 'Voyles2021: natCu(p,x)62Zn/natCu(p,x)63Zn', color = 'c')
    plt.plot([E_Cu_foil[0]+E_Cu_foil_unc_plus[0], E_Cu_foil[0]+E_Cu_foil_unc_plus[0]], [0,4], color = 'c', linestyle = 'dashed')
    plt.plot([E_Cu_foil[0]-E_Cu_foil_unc_minus[0], E_Cu_foil[0]-E_Cu_foil_unc_minus[0]], [0,4], color = 'c', linestyle = 'dotted')

    plt.plot([E_Cu_foil[1], E_Cu_foil[1]], [0,4], label = 'Voyles2021: natCu(p,x)62Zn/natCu(p,x)63Zn', color = 'royalblue')
    plt.plot([E_Cu_foil[1]+E_Cu_foil_unc_plus[1], E_Cu_foil[1]+E_Cu_foil_unc_plus[1]], [0,4], color = 'royalblue', linestyle = 'dashed')
    plt.plot([E_Cu_foil[1]-E_Cu_foil_unc_minus[1], E_Cu_foil[1]-E_Cu_foil_unc_minus[1]], [0,4], color = 'royalblue', linestyle = 'dotted')

    plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.title('First compartment in the 25 MeV stack')
    plt.legend()
    plt.show()








    # Magne's data:_____________________________________________________________________________________________________________
    # reaction_list = ['natTi_46Sc', 'natTi_48V', 'natCu_56Co','natCu_62Zn', 'natCu_63Zn', 'natCu_65Zn']
    # reaction_list_exfor = ['natCu_62Zn', 'natCu_63Zn', 'natCu_65Zn']

    # beam = 'p'

    # E_iaea, xs_iaea, xs_unc_iaea = generate_xs_lists(beam, reaction_list)
    # E_exfor, E_unc_exfor, xs_exfor, xs_unc_exfor = get_xs_from_exfor_files(beam, reaction_list_exfor)

    # E = np.linspace(54,100, 100000)
    # E_exfor_linspace = np.linspace(54,100, 10000000)


    # xs_interpol_exfor_Cu_62Zn = interpol_xs_exfor(xs_exfor[0], E_exfor[0])

    # plt.plot(E_exfor_linspace, xs_interpol_exfor_Cu_62Zn(E_exfor_linspace), 'b-')
    # plt.plot(E_exfor[0], xs_exfor[0], 'ro')
    # plt.show()

    # xs_interpol_iaea_Cu_56Co = interpol_xs_iaea(xs_iaea[2], E_iaea[2])
    # xs_interpol_iaea_Cu_63Zn = interpol_xs_iaea(xs_iaea[4], E_iaea[4])
    # xs_interpol_iaea_Cu_65Zn = interpol_xs_iaea(xs_iaea[5], E_iaea[5])

    # ratio_Cu_63Zn_div_Cu_65Zn_iaea = xs_interpol_iaea_Cu_63Zn(E)/xs_interpol_iaea_Cu_65Zn(E)
    # ratio_Cu_63Zn_div_Cu_56Co_iaea = xs_interpol_iaea_Cu_63Zn(E)/xs_interpol_iaea_Cu_56Co(E)
    # ratio_Cu_56Co_div_Cu_65Zn_iaea = xs_interpol_iaea_Cu_56Co(E)/xs_interpol_iaea_Cu_65Zn(E)

    # ratio_Cu_63Zn_div_Cu_65Zn_exsp_list = [8.40/2.868, 10.90/3.257, 12.98/3.68, 12.29/4.05, 14.0/4.21, 16.3/4.39, 17.5/4.66, 17.9/4.79]  #highest energy first
    # ratio_Cu_63Zn_div_Cu_56Co_exsp_list = [8.40/10.31, 10.90/12.12, 12.98/12.68, 12.29/10.95, 14.0/7.66, 16.3/4.47, 17.5/2.405, 17.9/1.272]  #highest energy first
    # ratio_Cu_56Co_div_Cu_65Zn_exsp_list = [10.31/2.868, 12.12/3.257, 12.68/3.68, 10.95/4.05, 7.66/4.21, 4.47/4.39, 2.405/4.66, 1.272/4.79]  #highest energy first
    # E_true_list = [90.94, 79.03, 72.22, 66.81, 62.73, 59.73, 57.11, 55.21] #[MeV]


    # colors = ['deeppink', 'crimson', 'tomato', 'gold', 'lightgreen', 'mediumseagreen', 'deepskyblue', 'mediumpurple']
    # plt.subplot(221)
    # plt.plot(E, ratio_Cu_63Zn_div_Cu_65Zn_iaea, color = 'hotpink', label = 'IAEA')
    # for i in range(len(ratio_Cu_63Zn_div_Cu_65Zn_exsp_list)):
    #     plt.plot([E[0],E[-1]], [ratio_Cu_63Zn_div_Cu_65Zn_exsp_list[i], ratio_Cu_63Zn_div_Cu_65Zn_exsp_list[i]], color = colors[i], linestyle = 'dashed', label = f'E = {E_true_list[i]}')
    # plt.xlabel('Energy (MeV)')
    # plt.ylabel('Ratio')
    # plt.title('natCu(p,x)63Zn/natCu(p,x)65Zn')
    # plt.legend()

    # plt.subplot(222)
    # plt.plot(E, ratio_Cu_63Zn_div_Cu_56Co_iaea, color = 'hotpink', label = 'IAEA')
    # for i in range(len(ratio_Cu_63Zn_div_Cu_56Co_exsp_list)):
    #     plt.plot([E[0],E[-1]], [ratio_Cu_63Zn_div_Cu_56Co_exsp_list[i], ratio_Cu_63Zn_div_Cu_56Co_exsp_list[i]], color = colors[i], linestyle = 'dashed', label = f'E = {E_true_list[i]}')
    # plt.xlabel('Energy (MeV)')
    # plt.ylabel('Ratio')
    # plt.title('natCu(p,x)63Zn/natCu(p,x)56Co')
    # plt.legend()

    # plt.subplot(223)
    # plt.plot(E, ratio_Cu_56Co_div_Cu_65Zn_iaea, color = 'hotpink', label = 'IAEA')
    # for i in range(len(ratio_Cu_56Co_div_Cu_65Zn_exsp_list)):
    #     plt.plot([E[0],E[-1]], [ratio_Cu_56Co_div_Cu_65Zn_exsp_list[i], ratio_Cu_56Co_div_Cu_65Zn_exsp_list[i]], color = colors[i], linestyle = 'dashed', label = f'E = {E_true_list[i]}')
    # plt.xlabel('Energy (MeV)')
    # plt.ylabel('Ratio')
    # plt.title('natCu(p,x)56Co/natCu(p,x)65Zn')
    # plt.legend()
    # plt.show()


    

    
    #Hannah's data:____________________________________________________________________________________________________________

    # xs_interpol_iaea_Ni_61Cu = interpol_xs(xs_iaea[0], E_iaea[0])
    # xs_interpol_iaea_Ni_56Co = interpol_xs(xs_iaea[1], E_iaea[1])
    # xs_interpol_iaea_Ni_58Co = interpol_xs(xs_iaea[2], E_iaea[2])
    # xs_interpol_iaea_Cu_62Zn = interpol_xs(xs_iaea[3], E_iaea[3])
    # xs_interpol_iaea_Cu_63Zn = interpol_xs(xs_iaea[4], E_iaea[4])
    # xs_interpol_iaea_Cu_65Zn = interpol_xs(xs_iaea[5], E_iaea[5])

    # ratio_Ni_61Cu_div_Ni_56Co_iaea = xs_interpol_iaea_Ni_61Cu(E)/xs_interpol_iaea_Ni_56Co(E)
    # ratio_Ni_61Cu_div_Ni_58Co_iaea = xs_interpol_iaea_Ni_61Cu(E)/xs_interpol_iaea_Ni_58Co(E)
    # ratio_Ni_58Co_div_Ni_56Co_iaea = xs_interpol_iaea_Ni_58Co(E)/xs_interpol_iaea_Ni_56Co(E)
    # ratio_Cu_62Zn_div_Cu_63Zn_iaea = xs_interpol_iaea_Cu_62Zn(E)/xs_interpol_iaea_Cu_63Zn(E)
    # ratio_Cu_62Zn_div_Cu_65Zn_iaea = xs_interpol_iaea_Cu_62Zn(E)/xs_interpol_iaea_Cu_65Zn(E)
    # ratio_Cu_65Zn_div_Cu_63Zn_iaea = xs_interpol_iaea_Cu_65Zn(E)/xs_interpol_iaea_Cu_63Zn(E)

    # ratio_Ni_61Cu_div_Ni_56Co_exsp = (A0[0]/(1-np.exp(-decay_const[0]*t_irr))) / (A0[1]/(1-np.exp(-decay_const[1]*t_irr)))
    # ratio_Ni_61Cu_div_Ni_58Co_exsp = (A0[0]/(1-np.exp(-decay_const[0]*t_irr))) / (A0[2]/(1-np.exp(-decay_const[2]*t_irr)))
    # ratio_Ni_58Co_div_Ni_56Co_exsp = (A0[2]/(1-np.exp(-decay_const[2]*t_irr))) / (A0[1]/(1-np.exp(-decay_const[1]*t_irr)))
    # ratio_Cu_62Zn_div_Cu_63Zn_exsp = (A0[3]/(1-np.exp(-decay_const[3]*t_irr))) / (A0[4]/(1-np.exp(-decay_const[4]*t_irr)))
    # ratio_Cu_62Zn_div_Cu_65Zn_exsp = (A0[3]/(1-np.exp(-decay_const[3]*t_irr))) / (A0[5]/(1-np.exp(-decay_const[5]*t_irr)))
    # ratio_Cu_65Zn_div_Cu_63Zn_exsp = (A0[5]/(1-np.exp(-decay_const[5]*t_irr))) / (A0[4]/(1-np.exp(-decay_const[4]*t_irr)))


    # plt.subplot(121)
    # plt.plot(E, ratio_Ni_61Cu_div_Ni_56Co_iaea, color = 'lightskyblue', label = 'natNi_61Cu/natNi_56Co')
    # plt.plot(E, ratio_Ni_61Cu_div_Ni_58Co_iaea, color = 'deepskyblue', label = 'natNi_61Cu/natNi_58Co')
    # plt.plot(E, ratio_Ni_58Co_div_Ni_56Co_iaea, color = 'cornflowerblue', label = 'natNi_58Co/natNi_56Co')
    # plt.plot([10,50], [ratio_Ni_61Cu_div_Ni_56Co_exsp, ratio_Ni_61Cu_div_Ni_56Co_exsp], color = 'lightskyblue', linestyle = 'dashed', label = 'natNi_61Cu/natNi_56Co')
    # plt.plot([10,50], [ratio_Ni_61Cu_div_Ni_58Co_exsp, ratio_Ni_61Cu_div_Ni_58Co_exsp], color = 'deepskyblue', linestyle = 'dashed', label = 'natNi_61Cu/natNi_58Co')
    # plt.plot([10,50], [ratio_Ni_58Co_div_Ni_56Co_exsp, ratio_Ni_58Co_div_Ni_56Co_exsp], color = 'cornflowerblue', linestyle = 'dashed', label = 'natNi_58Co/natNi_56Co')
    # plt.xlabel('Energy (MeV)')
    # plt.ylabel('Ratio')
    # plt.legend()
    # plt.subplot(122)
    # plt.plot(E, ratio_Cu_62Zn_div_Cu_63Zn_iaea, color = 'lightpink', label = 'natCu_62Zn/natCu_63Zn')
    # plt.plot(E, ratio_Cu_62Zn_div_Cu_65Zn_iaea, color = 'hotpink', label = 'natCu_62Zn/natCu_65Zn')
    # plt.plot(E, ratio_Cu_65Zn_div_Cu_63Zn_iaea, color = 'palevioletred', label = 'natCu_65Zn/natCu_63Zn')
    # plt.plot([10,50], [ratio_Cu_62Zn_div_Cu_63Zn_exsp, ratio_Cu_62Zn_div_Cu_63Zn_exsp], color = 'lightpink', linestyle = 'dashed', label = 'natCu_62Zn/natCu_63Zn')
    # plt.plot([10,50], [ratio_Cu_62Zn_div_Cu_65Zn_exsp, ratio_Cu_62Zn_div_Cu_65Zn_exsp], color = 'hotpink', linestyle = 'dashed', label = 'natCu_62Zn/natCu_65Zn')
    # plt.plot([10,50], [ratio_Cu_65Zn_div_Cu_63Zn_exsp, ratio_Cu_65Zn_div_Cu_63Zn_exsp], color = 'palevioletred', linestyle = 'dashed', label = 'natCu_65Zn/natCu_63Zn')
    # plt.xlabel('Energy (MeV)')
    # plt.ylabel('Ratio')
    # plt.legend()
    # plt.show()

















