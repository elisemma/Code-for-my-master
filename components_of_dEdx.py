from statistics import variance
import curie as ci
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import glob
from scipy.interpolate import CubicSpline
import seaborn as sns

    



def S_nucl(eng, z1, m1, z_el, m_el): #m i amu
    RM = (m1+m_el)*np.sqrt((z1**(2/3.0)+z_el**(2/3.0)))
    ER = 32.53*m_el*1E3*eng/(z1*z_el*RM)

    return (0.5*np.log(1.0+ER)/(ER+0.10718+ER**0.37544))*8.462*z1*z_el*m1/RM



def S_p(eng, m1, A):
        S = np.zeros(len(eng), dtype=np.float64) if eng.shape else np.array(0.0)
        E = 1E3*eng/m1

        beta_sq = np.where(E>=1E3, 1.0-1.0/(1.0+E/931478.0)**2, 0.9)
        B0 = np.where(E>=1E3, np.log(A[6]*beta_sq/(1.0-beta_sq))-beta_sq, 0.0)
        Y = np.log(E[(E>=1E3)&(E<=5E4)])
        B0[np.where((E>=1E3)&(E<=5E4), B0, 0)!=0] -= A[7]+A[8]*Y+A[9]*Y**2+A[10]*Y**3+A[11]*Y**4

        S[E>=1E3] = (A[5]/beta_sq[E>=1E3])*B0[E>=1E3]

        S_low = A[1]*E[(E>=10)&(E<1E3)]**0.45
        S_high = (A[2]/E[(E>=10)&(E<1E3)])*np.log(1.0+(A[3]/E[(E>=10)&(E<1E3)])+A[4]*E[(E>=10)&(E<1E3)])

        S[(E>=10)&(E<1E3)] = S_low*S_high/(S_low+S_high)
        S[(E>0)&(E<10)] = A[0]*E[(E>0)&(E<10)]**0.5

        return S, B0



def plot_S():
    el_Fe = ci.Element('Fe')
    el_Al = ci.Element('Al')
    el_Th = ci.Element('Th')
    el_Cu = ci.Element('Cu')

    el_Fe.plot_S(particle ='p')
    el_Al.plot_S(particle ='p')
    el_Th.plot_S(particle ='p')
    el_Cu.plot_S(particle ='p')



def plot_components(particle, energy):
    Z_p, mass_p = self._parse_particle(particle)
    S_nuc = self._S_nucl(energy, Z_p, mass_p)





if __name__ == '__main__': #______________________________________________________________________________________________________________________
    # m1 = 1.00727647 #proton  [u]
    m1 = 2.013553212745 #deuteron [u]
    z1 = 1
    eng = np.linspace(0,35,1000000) #[MeV]

    m_Fe = 55.845 #[u]
    z_Fe = 26
    A_Fe = np.array([3.519, 3.963, 6065.0, 1243.0, 0.007782, 0.01326, 3650, -9.809, 3.763, -0.5164, 0.0305, -0.00066])

    m_Al = 26.9815 #[u]
    z_Al = 13
    A_Al = np.array([4.154, 4.739, 2766.0, 164.5, 0.02023, 0.006628, 6309, -6.061, 2.46, -0.3535, 0.02173, -0.0004871])

    m_Cu = 63.546 #[u]
    z_Cu = 29
    A_Cu = np.array([3.696, 4.175, 4673.0, 387.8, 0.02188, 0.01479, 3174, -11.18, 4.252, -0.5791, 0.03399, -0.0007314])

    m_Th = 232.038 #[u]
    z_Th = 90
    A_Th = np.array([7.71, 8.679, 18830.0, 586.3, 0.002641, 0.04589, 1239, -20.04, 6.967, -0.8741, 0.04752, -0.0009516])

    

    # plot_S()
    # plot_components('p', np.linspace(0,40,1000000))
    S_nuc_Fe = S_nucl(eng, z1, m1, z_Fe, m_Fe)
    Sp_Fe, B0_Fe = S_p(eng, m1, A_Fe)

    S_nuc_Al = S_nucl(eng, z1, m1, z_Al, m_Al)
    Sp_Al, B0_Al = S_p(eng, m1, A_Al)

    S_nuc_Cu = S_nucl(eng, z1, m1, z_Cu, m_Cu)
    Sp_Cu, B0_Cu = S_p(eng, m1, A_Cu)

    S_nuc_Cu = S_nucl(eng, z1, m1, z_Cu, m_Cu)
    Sp_Cu, B0_Cu = S_p(eng, m1, A_Cu)

    S_nuc_Th = S_nucl(eng, z1, m1, z_Th, m_Th)
    Sp_Th, B0_Th = S_p(eng, m1, A_Th)


    plt.subplot(221)
    plt.plot(eng, S_nuc_Fe, label = 'S_nuc', color = 'lightpink')
    plt.plot(eng, Sp_Fe, label = 'S_p', color = 'hotpink')
    plt.plot(eng, B0_Fe, label = 'beta_0', color = 'violet')
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel('-dE/dx [MeV/cm]')
    plt.title('Fe')
    plt.subplot(222)
    plt.plot(eng, S_nuc_Al, label = 'S_nuc', color = 'lightpink')
    plt.plot(eng, Sp_Al, label = 'S_p', color = 'hotpink')
    plt.plot(eng, B0_Al, label = 'beta_0', color = 'violet')
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel('-dE/dx [MeV/cm]')
    plt.title('Al')
    plt.subplot(223)
    plt.plot(eng, S_nuc_Cu, label = 'S_nuc', color = 'lightpink')
    plt.plot(eng, Sp_Cu, label = 'S_p', color = 'hotpink')
    plt.plot(eng, B0_Cu, label = 'beta_0', color = 'violet')
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel('-dE/dx [MeV/cm]')
    plt.title('Cu')
    plt.subplot(224)
    plt.plot(eng, S_nuc_Th, label = 'S_nuc', color = 'lightpink')
    plt.plot(eng, Sp_Th, label = 'S_p', color = 'hotpink')
    plt.plot(eng, B0_Th, label = 'beta_0', color = 'violet')
    plt.legend()
    plt.xlabel('Energy [MeV]')
    plt.ylabel('-dE/dx [MeV/cm]')
    plt.title('Th')
    plt.show()
   







