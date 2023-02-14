from statistics import variance
import curie as ci
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import glob
from scipy.interpolate import CubicSpline
import seaborn as sns



def generate_xs_lists(reaction_list):
    E_xs_list = []
    xs_list = []
    xs_unc_list = []

    for reaction in reaction_list:
        with open('xs_txt/'+reaction+'.txt') as file:
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

            E_xs_list.append(E_reaction_list)
            xs_list.append(xs_reaction_list)
            xs_unc_list.append(xs_unc_reaction_list)
            file.close()

    return E_xs_list, xs_list, xs_unc_list



def interpol_xs(xs_list, E_xs_list):
    cs = CubicSpline(E_xs_list, xs_list)

    return cs




if __name__=='__main__': #______________________________________________________________________________________________________

    reaction_list = ['natNi_61Cu', 'natNi_56Co', 'natNi_58Co','natCu_62Zn', 'natCu_63Zn', 'natCu_65Zn']

    E_iaea, xs_iaea, xs_unc_iaea = generate_xs_lists(reaction_list)

    E = np.linspace(10,50,100000)

    xs_interpol_iaea_Ni_61Cu = interpol_xs(xs_iaea[0], E_iaea[0])
    xs_interpol_iaea_Ni_56Co = interpol_xs(xs_iaea[1], E_iaea[1])
    xs_interpol_iaea_Ni_58Co = interpol_xs(xs_iaea[2], E_iaea[2])
    xs_interpol_iaea_Cu_62Zn = interpol_xs(xs_iaea[3], E_iaea[3])
    xs_interpol_iaea_Cu_63Zn = interpol_xs(xs_iaea[4], E_iaea[4])
    xs_interpol_iaea_Cu_65Zn = interpol_xs(xs_iaea[5], E_iaea[5])

    ratio_Ni_61Cu_div_Ni_56Co = xs_interpol_iaea_Ni_61Cu(E)/xs_interpol_iaea_Ni_56Co(E)
    ratio_Ni_61Cu_div_Ni_58Co = xs_interpol_iaea_Ni_61Cu(E)/xs_interpol_iaea_Ni_58Co(E)
    ratio_Ni_58Co_div_Ni_56Co = xs_interpol_iaea_Ni_58Co(E)/xs_interpol_iaea_Ni_56Co(E)
    ratio_Cu_62Zn_div_Cu_63Zn = xs_interpol_iaea_Cu_62Zn(E)/xs_interpol_iaea_Cu_63Zn(E)
    ratio_Cu_62Zn_div_Cu_65Zn = xs_interpol_iaea_Cu_62Zn(E)/xs_interpol_iaea_Cu_65Zn(E)
    ratio_Cu_65Zn_div_Cu_63Zn = xs_interpol_iaea_Cu_65Zn(E)/xs_interpol_iaea_Cu_63Zn(E)


    plt.subplot(121)
    plt.plot(E, ratio_Ni_61Cu_div_Ni_56Co, color = 'lightskyblue', label = 'natNi_61Cu/natNi_56Co')
    plt.plot(E, ratio_Ni_61Cu_div_Ni_58Co, color = 'deepskyblue', label = 'natNi_61Cu/natNi_58Co')
    plt.plot(E, ratio_Ni_58Co_div_Ni_56Co, color = 'cornflowerblue', label = 'natNi_58Co/natNi_56Co')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.legend()
    plt.subplot(122)
    plt.plot(E, ratio_Cu_62Zn_div_Cu_63Zn, color = 'lightpink', label = 'natCu_62Zn/natCu_63Zn')
    plt.plot(E, ratio_Cu_62Zn_div_Cu_65Zn, color = 'hotpink', label = 'natCu_62Zn/natCu_65Zn')
    plt.plot(E, ratio_Cu_65Zn_div_Cu_63Zn, color = 'palevioletred', label = 'natCu_65Zn/natCu_63Zn')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.legend()
    plt.show()
















