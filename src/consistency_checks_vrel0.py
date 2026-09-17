"""
Zero-relative-velocity consistency checks for YAH (2017).

Loads the original and v_rel = 0 HyRec outputs, makes the six figures used in
the consistency-check note, and compares the 2017 local-feedback source
prescription with the printed Eq. (66).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HYREC = Path(__file__).resolve().parent / "HyRec_2017"
def load(f): return np.loadtxt(HYREC / f)

mdot_base, mdot_v0 = load("mdot_pbh.dat"), load("vrel0_mdot_pbh.dat")
L_base, L_v0 = load("L_pbh.dat"), load("vrel0_L_pbh.dat")
xe_base, xe_v0 = load("xe_pbh.dat"), load("vrel0_xe_pbh.dat")
vel_v0, Ts_v0 = load("vrel0_velocities.dat"), load("vrel0_Ts_pbh.dat")
gamma_v0, feedback_v0 = load("vrel0_gamma_pbh.dat"), load("vrel0_T_feedback.dat")

# columns used in the YAH output tables
masses = [1.0, 100.0, 10000.0]
mcols = {1.0: 1, 100.0: 2, 10000.0: 3}
ccols = {1.0: 1, 100.0: 3, 10000.0: 5}
pcols = {1.0: 2, 100.0: 4, 10000.0: 6}
mlabs = {1.0: r"$1\,M_\odot$", 100.0: r"$100\,M_\odot$", 10000.0: r"$10^4\,M_\odot$"}

# redshift range used in the note
z_all = mdot_v0[:, 0]
mask = (z_all >= 50) & (z_all <= 1e4)
z = z_all[mask]

# cgs constants
G, c = 6.67430e-8, 2.99792458e10
mp, me = 1.67262192369e-24, 9.1093837015e-28
sigmaT, alpha, kB = 6.6524587321e-25, 1 / 137.035999084, 1.380649e-16
Msun, Mpc = 1.98847e33, 3.0856775814913673e24

# cosmology used by the 2017 calculation
h, T0, omega_c, omh2, Nnueff = 0.7, 2.73, 0.120, 0.142, 3.046
rho_crit_h2 = 3 * (1e7 / Mpc)**2 / (8 * np.pi * G)
orh2 = 4.48162687719e-7 * T0**4 * (1 + 0.227107317660239 * Nnueff)
odeh2 = h**2 - omh2 - orh2

def Hubble(z):
    return 3.2407793e-18 * np.sqrt(omh2 * (1 + z)**3 + odeh2 + orh2 * (1 + z)**4)

def Jff(X):
    low = 4 / np.pi * np.sqrt(2 / (np.pi * X)) * (1 + 5.5 * X**1.25)
    high = 13.5 / np.pi * (np.log(2 * X * 0.56146 + 0.08) + 4 / 3)
    return np.where(X < 1, low, high)

def finish_plot(fname, ylabel, title, order=None, **kwargs):
    plt.xlabel("Redshift $z$"); plt.ylabel(ylabel); plt.title(title, pad=kwargs.pop("title_pad", 8))
    h, l = plt.gca().get_legend_handles_labels()
    if order is not None: h, l = [h[i] for i in order], [l[i] for i in order]
    plt.legend(h, l, frameon=False, **kwargs)
    plt.tight_layout(); plt.savefig(f"{fname}.png", dpi=220, bbox_inches="tight"); plt.show()


# figure 1: original accretion rate vs. v_rel = 0
plt.figure(figsize=(7.7, 4.8))
for M in masses:
    j = mcols[M]
    plt.loglog(z, mdot_base[mask, j], "--", label=f"{mlabs[M]}, original")
    plt.loglog(z, mdot_v0[mask, j], label=f"{mlabs[M]}, $v_{{\\rm rel}}=0$")

# reorder the two-row legend so the masses read left to right
finish_plot("fig1_mdot_comparison", r"$\dot m$", r"Accretion rate: original vs. $v_{\rm rel}=0$",
            order=[0, 3, 1, 4, 2, 5], ncol=3, fontsize=8.7, loc="lower center",
            bbox_to_anchor=(0.5, 1.01), title_pad=48)


# figure 2: accretion-rate enhancement
plt.figure(figsize=(7.4, 4.7))
for M in masses:
    j = mcols[M]
    plt.semilogx(z, mdot_v0[mask, j] / mdot_base[mask, j], label=mlabs[M])

plt.axhline(1, ls=":", lw=1)
finish_plot("fig2_mdot_enhancement", r"$\dot m(0)/\dot m(v_{\rm eff})$",
            "Increase in accretion when relative motion is removed", fontsize=10.5)


# figure 3: main consistency checks for 10^4 solar masses
M, fPBH = 1e4, 1.0
vB = vel_v0[mask, 2]
GM = G * M * Msun
rB, tB = GM / vB**2, GM / vB**3

rho_dm = omega_c * rho_crit_h2 * (1 + z)**3
rsep = (3 * M * Msun / (4 * np.pi * fPBH * rho_dm))**(1 / 3)
isolated_ratio = rB / rsep
steady_ratio = Hubble(z) * tB

mdot = mdot_v0[mask, mcols[M]]
Ts = Ts_v0[mask, ccols[M]]
cooling_ratio = mdot * alpha * Jff(kB * Ts / (me * c**2))

# recover physical Mdot only for the near-horizon density estimate
Mdot = mdot * 1.4e17 * M
rS = 2 * GM / c**2
neS = Mdot / (4 * np.pi * mp * rS**2 * c)
thomson_proxy = rS * neS * sigmaT

plt.figure(figsize=(7.4, 4.7))
plt.loglog(z, isolated_ratio, label=r"$r_B/\bar r_{\rm PBH}$")
plt.loglog(z, steady_ratio, label=r"$H t_B$")
plt.loglog(z, cooling_ratio, label=r"$t_{\rm acc}/t_{\rm ff}$")
plt.loglog(z, thomson_proxy, label=r"$\mathcal{C}_{\rm T,S}$")
plt.axhline(1, ls=":", lw=1, label="unity")

finish_plot("fig3_consistency_1e4", "Dimensionless ratio",
            r"Consistency checks for $10^4\,M_\odot$ at $v_{\rm rel}=0$", fontsize=9)

# figure 4: luminosity enhancement
plt.figure(figsize=(7.4, 4.7))
for M in masses:
    plt.semilogx(z, L_v0[mask, ccols[M]] / L_base[mask, ccols[M]], label=mlabs[M])

plt.axhline(1, ls=":", lw=1)
finish_plot("fig4_luminosity_enhancement", r"$L(0)/\langle L\rangle$",
            "Luminosity increase in the collisional-ionization branch", fontsize=10)

# figure 5: global HyRec ionization response
xb = (xe_base[:, 0] >= 50) & (xe_base[:, 0] <= 2000)
xv = (xe_v0[:, 0] >= 50) & (xe_v0[:, 0] <= 2000)

plt.figure(figsize=(7.4, 4.7))
plt.loglog(xe_base[xb, 0], xe_base[xb, 1], label="No PBHs")
plt.loglog(xe_base[xb, 0], xe_base[xb, 3], "--", label=r"$100\,M_\odot$, original")
plt.loglog(xe_v0[xv, 0], xe_v0[xv, 3], label=r"$100\,M_\odot$, $v_{\rm rel}=0$")

finish_plot("fig5_xe_response", r"Free-electron fraction $x_e$",
            r"HyRec ionization history for the $100\,M_\odot$ example", fontsize=9.5)

# figure 6: local feedback, source prescription vs. printed Eq. 66
M = 1e4
xe_fb = np.minimum(xe_base[mask, 1], 1.0)
vB_fb = vel_v0[mask, 2]
Tcmb = T0 * (1 + z)

def eq66_feedback(M, col):
    gamma = gamma_v0[mask, mcols[M]]
    return (0.07 * xe_fb / (1 + xe_fb) * L_v0[mask, col] * vB_fb / c
            * mp * c**2 / (kB * Tcmb) * np.sqrt(1 + gamma**(2 / 3)))

plt.figure(figsize=(7.6, 4.8))
for col, branch in [(ccols[M], "Collisional"), (pcols[M], "Photoionization")]:
    plt.loglog(z, feedback_v0[mask, col], "--", label=f"{branch}: 2017 source")
    plt.loglog(z, eq66_feedback(M, col), label=f"{branch}: printed Eq. (66)")

plt.axhline(1, ls=":", lw=1)
finish_plot("fig6_local_feedback", r"Local feedback diagnostic $\mathcal{F}_{\rm fb}$",
            r"Local thermal feedback for $10^4\,M_\odot$ at $v_{\rm rel}=0$",
            ncol=2, fontsize=8.8)

# values quoted in the note
print("\nMaximum accretion enhancements:")
for M in masses:
    ratio = mdot_v0[mask, mcols[M]] / mdot_base[mask, mcols[M]]
    print(f"{M:g} Msun: {ratio.max():.2f}")

print("\n10^4 Msun consistency-check maxima:")
print(f"rB/rsep = {isolated_ratio.max():.3f}")
print(f"H tB = {steady_ratio.max():.3f}")
print(f"tacc/tff = {cooling_ratio.max():.3f}")
print(f"Thomson column = {thomson_proxy.max():.3f}")
print("\nLocal-feedback maxima:")
for M in masses:
    for col, branch in [(ccols[M], "coll"), (pcols[M], "photo")]:
        source = feedback_v0[mask, col]
        printed = eq66_feedback(M, col)
        print(f"{M:g} Msun {branch}: source={source.max():.3g}, Eq66={printed.max():.3g}")