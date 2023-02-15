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




if __name__=='__main__': #______________________________________________________________________________________________________

    reaction_list = ['natTi_46Sc', 'natTi_48V', 'natCu_56Co','natCu_62Zn', 'natCu_63Zn', 'natCu_65Zn']
    reaction_list_exfor = ['natCu_62Zn', 'natCu_63Zn', 'natCu_65Zn']

    beam = 'p'

    E_iaea, xs_iaea, xs_unc_iaea = generate_xs_lists(beam, reaction_list)
    E_exfor, E_unc_exfor, xs_exfor, xs_unc_exfor = get_xs_from_exfor_files(beam, reaction_list_exfor)

    E = np.linspace(54,100, 100000)
    E_exfor_linspace = np.linspace(54,100, 10000000)


    xs_interpol_exfor_Cu_62Zn = interpol_xs_exfor(xs_exfor[0], E_exfor[0])

    plt.plot(E_exfor_linspace, xs_interpol_exfor_Cu_62Zn(E_exfor_linspace), 'b-')
    plt.plot(E_exfor[0], xs_exfor[0], 'ro')
    plt.show()

    xs_interpol_iaea_Cu_56Co = interpol_xs_iaea(xs_iaea[2], E_iaea[2])
    xs_interpol_iaea_Cu_63Zn = interpol_xs_iaea(xs_iaea[4], E_iaea[4])
    xs_interpol_iaea_Cu_65Zn = interpol_xs_iaea(xs_iaea[5], E_iaea[5])

    ratio_Cu_63Zn_div_Cu_65Zn_iaea = xs_interpol_iaea_Cu_63Zn(E)/xs_interpol_iaea_Cu_65Zn(E)
    ratio_Cu_63Zn_div_Cu_56Co_iaea = xs_interpol_iaea_Cu_63Zn(E)/xs_interpol_iaea_Cu_56Co(E)
    ratio_Cu_56Co_div_Cu_65Zn_iaea = xs_interpol_iaea_Cu_56Co(E)/xs_interpol_iaea_Cu_65Zn(E)

    ratio_Cu_63Zn_div_Cu_65Zn_exsp_list = [8.40/2.868, 10.90/3.257, 12.98/3.68, 12.29/4.05, 14.0/4.21, 16.3/4.39, 17.5/4.66, 17.9/4.79]  #highest energy first
    ratio_Cu_63Zn_div_Cu_56Co_exsp_list = [8.40/10.31, 10.90/12.12, 12.98/12.68, 12.29/10.95, 14.0/7.66, 16.3/4.47, 17.5/2.405, 17.9/1.272]  #highest energy first
    ratio_Cu_56Co_div_Cu_65Zn_exsp_list = [10.31/2.868, 12.12/3.257, 12.68/3.68, 10.95/4.05, 7.66/4.21, 4.47/4.39, 2.405/4.66, 1.272/4.79]  #highest energy first
    E_true_list = [90.94, 79.03, 72.22, 66.81, 62.73, 59.73, 57.11, 55.21] #[MeV]




    colors = ['deeppink', 'crimson', 'tomato', 'gold', 'lightgreen', 'mediumseagreen', 'deepskyblue', 'mediumpurple']
    plt.subplot(221)
    plt.plot(E, ratio_Cu_63Zn_div_Cu_65Zn_iaea, color = 'hotpink', label = 'IAEA')
    for i in range(len(ratio_Cu_63Zn_div_Cu_65Zn_exsp_list)):
        plt.plot([E[0],E[-1]], [ratio_Cu_63Zn_div_Cu_65Zn_exsp_list[i], ratio_Cu_63Zn_div_Cu_65Zn_exsp_list[i]], color = colors[i], linestyle = 'dashed', label = f'E = {E_true_list[i]}')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.title('natCu(p,x)63Zn/natCu(p,x)65Zn')
    plt.legend()

    plt.subplot(222)
    plt.plot(E, ratio_Cu_63Zn_div_Cu_56Co_iaea, color = 'hotpink', label = 'IAEA')
    for i in range(len(ratio_Cu_63Zn_div_Cu_56Co_exsp_list)):
        plt.plot([E[0],E[-1]], [ratio_Cu_63Zn_div_Cu_56Co_exsp_list[i], ratio_Cu_63Zn_div_Cu_56Co_exsp_list[i]], color = colors[i], linestyle = 'dashed', label = f'E = {E_true_list[i]}')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.title('natCu(p,x)63Zn/natCu(p,x)56Co')
    plt.legend()

    plt.subplot(223)
    plt.plot(E, ratio_Cu_56Co_div_Cu_65Zn_iaea, color = 'hotpink', label = 'IAEA')
    for i in range(len(ratio_Cu_56Co_div_Cu_65Zn_exsp_list)):
        plt.plot([E[0],E[-1]], [ratio_Cu_56Co_div_Cu_65Zn_exsp_list[i], ratio_Cu_56Co_div_Cu_65Zn_exsp_list[i]], color = colors[i], linestyle = 'dashed', label = f'E = {E_true_list[i]}')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.title('natCu(p,x)56Co/natCu(p,x)65Zn')
    plt.legend()
    plt.show()


    

    

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

















