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

    xs_interpol_iaea_Ni_61Cu = interpol_xs(xs_iaea[0], E_iaea[0])
    xs_interpol_iaea_Ni_56Co = interpol_xs(xs_iaea[1], E_iaea[1])

    E = np.linspace(5,50,100000)

    ratio = xs_interpol_iaea_Ni_61Cu(E)/xs_interpol_iaea_Ni_56Co(E)


    plt.plot(E, ratio, color = 'hotpink', label = 'natNi_61Cu/natNi_56Co')
    # plt.plot(E, xs_interpol_iaea_Ni_61Cu(E), color = 'skyblue', label = 'xs for natNi_61Cu')
    # plt.plot(E, xs_interpol_iaea_Ni_56Co(E), color = 'violet', label = 'xs for natNi_56Co')
    # plt.xlabel('Energy (MeV)')
    plt.ylabel('Ratio')
    plt.legend()
    plt.show()
















