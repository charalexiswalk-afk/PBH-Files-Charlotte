# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:19:45 2026

@author: chara
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data = np.loadtxt(
    r"C:\Users\chara\OneDrive\python\HYREC-2-master\HYREC-2-master\output_xe.dat"
)

z = data[:,0]
xe = data[:,1]
Tm = data[:,2]

mask = (z >= 100) & (z <= 8000)

z = z[mask]
xe = xe[mask]
Tm = Tm[mask]

G = 6.67430e-11
c = 2.99792458e8
sigma_T = 6.6524587321e-29
m_p = 1.67262192369e-27
m_e = 9.1093837015e-31
k_B = 1.380649e-23
M_sun = 1.98847e30
a_rad = 7.5657e-16
T0 = 2.7255

masses = [1, 1e2, 1e4]

labels = [
    r"$1\,M_\odot$",
    r"$10^2\,M_\odot$",
    r"$10^4\,M_\odot$"
]

colors = ["red", "purple", "blue"]

mec2_K = m_e * c**2 / k_B

def veff_calc(vB, vL):

    y = np.linspace(0,8,2000)

    P = (
        np.sqrt(2/np.pi)
        * 3**1.5
        * y**2
        * np.exp(-1.5*y**2)
    )

    P /= np.trapz(P, y)

    v = y[:,None] * vL[None,:]

    avg = np.trapz(
        P[:,None] * (vB[None,:]**2 + v**2)**(-3),
        y,
        axis=0
    )

    return avg**(-1/6)

def F(Y):

    return Y * (1 + Y/0.27)**(-1/3)

def temperature_s(gamma, xe, case):

    tau = 1.5 / (5 + gamma**(2/3))

    if case == "collisional":
        chi = (2/(1 + xe))**8
    else:
        chi = 1.0

    Ys = (
        chi**(-2/3)
        * (2/(1 + xe))
        * (tau/4)
        * (1 - 5*tau/2)**(1/3)
        * (m_p/m_e)
    )

    return mec2_K * F(Ys)

fig, ax = plt.subplots(figsize=(6,4.5))

for Mfac, label, color in zip(masses, labels, colors):

    M = Mfac * M_sun

    Tcmb = T0 * (1 + z)
    rho_cmb = a_rad * Tcmb**4

    vB = np.sqrt((1 + xe) * k_B * Tm / m_p)

    vL = np.minimum(1.0, z/1000.0) * 30000.0

    veff = veff_calc(vB, vL)

    tB = G * M / veff**3

    gamma = (
        8 * xe * sigma_T * rho_cmb * tB
        / (3 * m_e * c * (1 + xe))
    )

    Ts_coll = temperature_s(
        gamma,
        xe,
        "collisional"
    )

    Ts_photo = temperature_s(
        gamma,
        xe,
        "photoionization"
    )

    ax.loglog(
        z,
        Ts_coll,
        color=color,
        linewidth=2,
        label=label
    )

    ax.loglog(
        z,
        Ts_photo,
        color=color,
        linewidth=2,
        linestyle="--"
    )

ax.set_xlim(100, 10000)
ax.set_ylim(1e8, 1e12)

ax.set_xlabel(r"$z$", fontsize=12)
ax.set_ylabel(r"$T_S\ {\rm (K)}$", fontsize=12)

ax.grid(True, which="both", alpha=0.2)

mass_legend = ax.legend(
    frameon=False,
    loc="lower left"
)

style_lines = [
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=2,
        linestyle="-"
    ),

    Line2D(
        [0],
        [0],
        color="black",
        linewidth=2,
        linestyle="--"
    )
]

style_labels = [
    "Collisional ionization",
    "Photoionization"
]

style_legend = ax.legend(
    style_lines,
    style_labels,
    frameon=False,
    loc="upper right"
)

ax.add_artist(mass_legend)

plt.tight_layout()
plt.show()