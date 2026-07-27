from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

# background cosmology
H0 = 67.4 * 1000 / 3.085677581e22
Omega_b = 0.049
rho_crit0 = 3 * H0**2 / (8 * np.pi * G)
rho_b0 = Omega_b * rho_crit0

# accretion eigenvalue limits
lambda_ad = 0.25 * (3.0/5.0)**1.5
lambda_iso = 0.25 * np.exp(1.5)

# average over the relative-velocity distribution
def veff_calc(vB, vL):
    y = np.linspace(0, 8, 2000)
    P = np.sqrt(2 / np.pi) * 3**1.5 * y**2 * np.exp(-1.5 * y**2)
    P /= np.trapezoid(P, y)
    v = y[:, None] * vL[None, :]
    return np.trapezoid(P[:, None] * (vB[None, :]**2 + v**2)**(-3), y, axis=0)**(-1 / 6)

# include Compton cooling and drag
def lambda_total(gamma, beta):
    lam_gamma = lambda_ad + (lambda_iso - lambda_ad) * (gamma**2 / (88.0 + gamma**2))**0.22
    lam_beta = np.exp(4.5 / (3.0 + beta**0.75)) / (np.sqrt(1.0 + beta) + 1.0)**2
    return lam_gamma * lam_beta / lambda_iso

fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
masses, labels, colors = [1, 1e2, 1e4], [r"$1\,M_\odot$", r"$10^2\,M_\odot$", r"$10^4\,M_\odot$"], ["red", "purple", "blue"]

# calculate lambda and dimensionless accretion rate
for Mfac, label, color in zip(masses, labels, colors):
    M = Mfac * M_sun
    rho_cmb, rho_b = a_rad * (T0 * (1 + z))**4, rho_b0 * (1 + z)**3
    vB, vL = np.sqrt((1 + xe) * k_B * Tm / m_p), np.minimum(1.0, z / 1000.0) * 30000.0
    
    veff = veff_calc(vB, vL)
    tB = G * M / veff**3
    beta = (4 / 3) * xe * sigma_T * rho_cmb * tB / (m_p * c)
    gamma = 8 * xe * sigma_T * rho_cmb * tB / (3 * m_e * c * (1 + xe))
    
    lam = lambda_total(gamma, beta)
    Mdot = 4 * np.pi * rho_b * (G * M)**2 * lam / veff**3
    Ledd = 4 * np.pi * G * M * m_p * c / sigma_T
    mdot = Mdot * c**2 / Ledd

    axes[0].loglog(z, lam, color=color, lw=2, label=label)
    axes[1].loglog(z, mdot, color=color, lw=2, label=label)

# figure formatting
axes[0].axhline(lambda_iso, color="black", ls=":", lw=1)
axes[0].axhline(lambda_ad, color="black", ls=":", lw=1)

for ax in axes:
    ax.grid(True, which="both", alpha=0.2)
    ax.set(xlim=(100, 10000), xticks=[100, 200, 500, 1000, 2000, 5000, 10000],
           xticklabels=[r"$100$", r"$200$", r"$500$", r"$1000$", r"$2000$", r"$5000$", r"$10000$"])

axes[0].set(ylim=(0.05, 2), ylabel=r"$\lambda$")
axes[1].set(ylim=(1e-5, 10), ylabel=r"$\dot{m}$", xlabel=r"$z$")
axes[0].legend(frameon=False)

plt.tight_layout()
plt.show()
    
