from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# load HyRec recombination history
DATA_FILE = Path(__file__).resolve().parent.parent / "data/output_xe.dat"
data = np.loadtxt(DATA_FILE)

z = data[:,0]
xe = data[:,1]
Tm = data[:,2]

# restrict to the redshift range used in the figure
mask = (z >= 100) & (z <= 8000)
z = z[mask]
xe = xe[mask]
Tm = Tm[mask]

# physical constants
G = 6.67430e-11
c = 2.99792458e8
sigma_T = 6.6524587321e-29
m_p = 1.67262192369e-27
m_e = 9.1093837015e-31
k_B = 1.380649e-23
M_sun = 1.98847e30
a_rad = 7.5657e-16
T0 = 2.7255
mec2_K = m_e * c**2 / k_B

def veff_calc(vB, vL):
    y = np.linspace(0, 8, 2000)
    P = np.sqrt(2 / np.pi) * 3**1.5 * y**2 * np.exp(-1.5 * y**2)
    P /= np.trapezoid(P, y)
    v = y[:, None] * vL[None, :]
    return np.trapezoid(P[:, None] * (vB[None, :]**2 + v**2)**(-3), y, axis=0)**(-1 / 6)

# inner accretion-flow temperature
def temperature_s(gamma, xe, case):
    tau = 1.5 / (5 + gamma**(2 / 3))
    chi = (2 / (1 + xe))**8 if case == "collisional" else 1.0
    Ys = chi**(-2 / 3) * (2 / (1 + xe)) * (tau / 4) * (1 - 5 * tau / 2)**(1 / 3) * (m_p / m_e)
    return mec2_K * (Ys * (1 + Ys / 0.27)**(-1 / 3))

fig, ax = plt.subplots(figsize=(6, 4.5))
masses, labels, colors = [1, 1e2, 1e4], [r"$1\,M_\odot$", r"$10^2\,M_\odot$", r"$10^4\,M_\odot$"], ["red", "purple", "blue"]

# calculate the temperature branches
for Mfac, label, color in zip(masses, labels, colors):
    rho_cmb = a_rad * (T0 * (1 + z))**4
    vB, vL = np.sqrt((1 + xe) * k_B * Tm / m_p), np.minimum(1.0, z / 1000.0) * 30000.0
    veff = veff_calc(vB, vL)
    tB = G * (Mfac * M_sun) / veff**3
    gamma = 8 * xe * sigma_T * rho_cmb * tB / (3 * m_e * c * (1 + xe))

    ax.loglog(z, temperature_s(gamma, xe, "collisional"), color=color, lw=2, label=label)
    ax.loglog(z, temperature_s(gamma, xe, "photoionization"), color=color, lw=2, ls="--")

ax.set(xlim=(100, 10000), ylim=(1e8, 1e12), xlabel=r"$z$", ylabel=r"$T_S\ {\rm (K)}$")
ax.grid(True, which="both", alpha=0.2)

# legend
mass_legend = ax.legend(frameon=False, loc="lower left")
style_lines = [Line2D([0], [0], color="black", lw=2, ls="-"), Line2D([0], [0], color="black", lw=2, ls="--")]
ax.legend(style_lines, ["Collisional ionization", "Photoionization"], frameon=False, loc="upper right")
ax.add_artist(mass_legend)

plt.tight_layout()
plt.show()
