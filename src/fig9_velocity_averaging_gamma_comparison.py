"""
Author: Charlotte Walker

Description:
Reconstructs the Figure 9 thermal-feedback calculation from Ali-Haïmoud & Kamionkowski (2017). The luminosity is averaged over the 
distribution of relative baryon-dark matter velocities and combined with the thermal-feedback prescription used throughout the paper.

Equation (66) contains the factor sqrt(1 + gamma^(2/3)), whereas the original HyRec implementation evaluates the corresponding quantity
using 1 + gamma^(1/3).

The original HyRec implementation replaces the Eq. (66) expression with the interpolation 1 + gamma^(1/3), which has the same limiting 
behaviour for small and large gamma. This script compares the two prescriptions while keeping every other part of the calculation unchanged 
in order to isolate the effect of this approximation on the reconstructed Figure 9 curves.
"""

import io, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ZIP_PATH = Path(__file__).resolve().parent / "HyRec_2017.zip"
PREFIX = "HyRec_2017/"

def load_data(name):
    with zipfile.ZipFile(ZIP_PATH) as archive:
        return np.loadtxt(io.BytesIO(archive.read(PREFIX + name)))

# load HyRec outputs
feedback_data = load_data("T_feedback.dat")
beta_data = load_data("beta_pbh.dat")
gamma_data = load_data("gamma_pbh.dat")
velocity_data = load_data("velocities.dat")

# recover the gas quantities used in Figure 9
z = feedback_data[:,0]
xe = np.clip(3670.0 * beta_data[:,1] / gamma_data[:,1] - 1.0, 0.0, 1.0)
Tgas = (velocity_data[:,2] / 9.09e3)**2 / (1.0 + xe)
vrel_rms = velocity_data[:,1]
Trel = 1.21e-8 * vrel_rms**2 / (1.0 + xe)
Teff = np.where(Trel < Tgas, Tgas, np.sqrt(Tgas * Trel))

# accretion and luminosity functions
def beta_pbh(M, z, xe, Teff):
    vB = 9.09e3 * np.sqrt((1.0 + xe) * Teff)
    return 7.45e-24 * xe * (1.33e26 * M / vB**3) * (1.0 + z)**4

def gamma_pbh(M, z, xe, Teff):
    return 3.67e3 / (1.0 + xe) * beta_pbh(M, z, xe, Teff)

def lambda_pbh(M, z, xe, Teff):
    beta, gamma = beta_pbh(M, z, xe, Teff), gamma_pbh(M, z, xe, Teff)
    lam_ad, lam_iso = 0.6**1.5 / 4.0, np.exp(1.5) / 4.0
    lam_drag = np.exp(4.5 / (3.0 + beta**0.75)) / (np.sqrt(1.0 + beta) + 1.0)**2
    lam_nodrag = lam_ad + (lam_iso - lam_ad) * (gamma**2 / (88.0 + gamma**2))**0.22
    return lam_drag * lam_nodrag / lam_iso

def Mdot_pbh(M, z, xe, Teff):
    vB = 9.09e3 * np.sqrt((1.0 + xe) * Teff)
    return 9.15e22 * M**2 * ((1.0 + z) / vB)**3 * lambda_pbh(M, z, xe, Teff)

def TS_over_me(M, z, xe, Teff, collisional):
    gamma = gamma_pbh(M, z, xe, Teff)
    tau = 1.5 / (5.0 + gamma**(2.0 / 3.0))
    YS = (2.0 / (1.0 + xe)) * (tau / 4.0) * (1.0 - 2.5 * tau)**(1.0 / 3.0) * 1836.0
    if collisional: YS *= ((1.0 + xe) / 2.0)**8
    return YS / (1.0 + YS / 0.27)**(1.0 / 3.0)

def free_free_j(X):
    X = np.maximum(np.asarray(X), 1.0e-300)
    G = np.empty_like(X)
    low = X < 1.0
    G[low] = 4.0 / np.pi * np.sqrt(2.0 / (np.pi * X[low])) * (1.0 + 5.5 * X[low]**1.25)
    G[~low] = 13.5 / np.pi * (np.log(2.0 * X[~low] * 0.56146 + 0.08) + 4.0 / 3.0)
    return G

def L_pbh(M, z, xe, Teff, collisional):
    Mdot = Mdot_pbh(M, z, xe, Teff)
    X = TS_over_me(M, z, xe, Teff, collisional)
    eps_over_mdot = X / 1836.0 / 137.0 * free_free_j(X)
    return (Mdot / (1.4e17 * M)) * eps_over_mdot * Mdot * 9.0e20

# average luminosity over relative velocities
def average_luminosity(M, z, xe, Tgas, collisional):
    x = np.linspace(0.0, 5.0, 50)[:,None]
    weight = x**2 * np.exp(-1.5 * x**2)
    vrel = x * vrel_rms[None,:]
    T = Tgas[None,:] + 1.21e-8 * vrel**2 / (1.0 + xe[None,:])
    return np.sum(weight * L_pbh(M, z[None,:], xe[None,:], T, collisional), axis=0) / np.sum(weight, axis=0)

# thermal-feedback quantity from eq. (66)
def feedback_prefactor(xe, Teff):
    return np.sqrt(Teff / 1.21e-8) / 3.0e10 * 0.067 * xe / (1.0 + xe) * 1.1e13 / Teff

# compare eq. (66) with the approximation used in HyRec
def feedback_curves(M, collisional):
    Lavg = average_luminosity(M, z, xe, Tgas, collisional)
    gamma = gamma_pbh(M, z, xe, Teff)
    base = Lavg / (1.26e38 * M) * feedback_prefactor(xe, Teff)
    exact = base * np.sqrt(1.0 + gamma**(2.0 / 3.0))
    approx = base * (1.0 + gamma**(1.0 / 3.0))
    return exact, approx, gamma

fig, ax = plt.subplots(figsize=(8.2, 5.9))
masses, labels, colors = [1, 1e2, 1e4], [r"$1\,M_\odot$", r"$10^2\,M_\odot$", r"$10^4\,M_\odot$"], ["red", "purple", "blue"]

# plot both ionization branches for each PBH mass
for M, color, label in zip(masses, colors, labels):
    for collisional, ls in ((True, "-"), (False, "--")):
        exact, approx, gamma = feedback_curves(M, collisional)
        ax.loglog(z, exact, color=color, ls=ls, lw=2.2, label=label if collisional else None)
        ax.loglog(z, approx, color=color, ls=ls, lw=1.1, marker="o", ms=2.5, markevery=65, mfc="white")

    ratio = (1.0 + gamma**(1.0 / 3.0)) / np.sqrt(1.0 + gamma**(2.0 / 3.0))
    print(f"M={M:g} M_sun: approximation overestimates by {100 * (ratio.max() - 1):.1f}% at most")

ax.axhline(1.0, color="black", lw=1.0, ls=":")
ax.set(xlim=(3.0e2, 2.0e4), ylim=(1.0e-8, 1.0e2), xlabel=r"$z$",
       ylabel=r"$\max\!\left(\dot T_{\mathrm{Compt},L}/\dot T\right)$",
       title="Figure 9 thermal feedback with two $\gamma$ prescriptions")
ax.grid(True, which="both", alpha=0.2)

mass_legend = ax.legend(loc="lower right", frameon=False, title="PBH mass")
ax.add_artist(mass_legend)

style_handles = [
    Line2D([0], [0], color="black", lw=2.2, label=r"$\sqrt{1+\gamma^{2/3}}$"),
    Line2D([0], [0], color="black", lw=1.1, marker="o", mfc="white", label=r"$1+\gamma^{1/3}$"),
    Line2D([0], [0], color="black", lw=2.2, ls="-", label="Collisional"),
    Line2D([0], [0], color="black", lw=2.2, ls="--", label="Photoionization"),
]
ax.legend(handles=style_handles, loc="upper right", frameon=False, fontsize=8)

fig.tight_layout()
plt.show()
