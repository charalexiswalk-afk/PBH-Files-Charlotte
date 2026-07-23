import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "output_xe.dat"

data = np.loadtxt(DATA_FILE)

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
labels = [r"$1\,M_\odot$", r"$10^2\,M_\odot$", r"$10^4\,M_\odot$"]
colors = ["red", "purple", "blue"]

def veff_calc(vB, vL):

    y = np.linspace(0,8,2000)

    P = np.sqrt(2/np.pi) * 3**1.5 * y**2 * np.exp(-1.5*y**2)
    P /= np.trapz(P, y)

    v = y[:,None] * vL[None,:]

    avg = np.trapz(
        P[:,None] * (vB[None,:]**2 + v**2)**(-3),
        y,
        axis=0
    )

    return avg**(-1/6)

fig, axes = plt.subplots(2,1,figsize=(6,7),sharex=True)

for Mfac, label, color in zip(masses, labels, colors):

    M = Mfac * M_sun

    Tcmb = T0 * (1 + z)
    rho_cmb = a_rad * Tcmb**4

    vB = np.sqrt((1 + xe) * k_B * Tm / m_p)
    vL = np.minimum(1.0, z/1000.0) * 30000.0

    veff = veff_calc(vB, vL)

    tB = G * M / veff**3

    beta = (4/3) * xe * sigma_T * rho_cmb * tB / (m_p * c)

    gamma = (
        8 * xe * sigma_T * rho_cmb * tB
        / (3 * m_e * c * (1 + xe))
    )

    axes[0].loglog(z, beta, color=color, linewidth=2, label=label)
    axes[1].loglog(z, gamma, color=color, linewidth=2, label=label)

for ax in axes:
    ax.axhline(1, color="black", linewidth=1, alpha=0.5)
    ax.grid(True, which="both", alpha=0.2)

    ax.set_xlim(100, 10000)
    ax.set_ylim(1e-3, 1e3)

    ax.set_xticks([100, 200, 500, 1000, 2000, 5000, 10000])
    ax.set_xticklabels(
        [r"$100$", r"$200$", r"$500$", r"$1000$",
         r"$2000$", r"$5000$", r"$10000$"]
    )

axes[0].set_ylabel(r"$\beta$", fontsize=12)
axes[1].set_ylabel(r"$\gamma$", fontsize=12)
axes[1].set_xlabel(r"$z$", fontsize=12)

axes[0].legend(frameon=False)

plt.tight_layout()
plt.show()
