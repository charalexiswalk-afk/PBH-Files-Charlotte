import io, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ZIP_PATH = Path(__file__).resolve().parent / "HyRec_2017.zip"
PREFIX = "HyRec_2017/"

def load_dat_from_zip(zip_path: Path, member: str) -> np.ndarray:
    """Read a whitespace-delimited data file from the HyRec archive."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        return np.loadtxt(io.BytesIO(archive.read(member)))

# Load data files
feedback_data = load_dat_from_zip(ZIP_PATH, PREFIX + "T_feedback.dat")
luminosity_data = load_dat_from_zip(ZIP_PATH, PREFIX + "L_pbh.dat")
beta_data = load_dat_from_zip(ZIP_PATH, PREFIX + "beta_pbh.dat")
gamma_data = load_dat_from_zip(ZIP_PATH, PREFIX + "gamma_pbh.dat")
velocity_data = load_dat_from_zip(ZIP_PATH, PREFIX + "velocities.dat")

z = feedback_data[:, 0]
xe = np.clip(3670.0 * beta_data[:, 1] / gamma_data[:, 1] - 1.0, 0.0, 1.0)
Tgas = (velocity_data[:, 2] / 9.09e3) ** 2 / (1.0 + xe)
Teff_original = (velocity_data[:, 3] / 9.09e3) ** 2 / (1.0 + xe)

def beta_pbh(M, z, xe, Teff):
    return 7.45e-24 * xe * (1.33e26 * M / (9.09e3 * np.sqrt((1.0 + xe) * Teff))**3) * (1.0 + z)**4

def gamma_pbh(M, z, xe, Teff):
    return 3.67e3 / (1.0 + xe) * beta_pbh(M, z, xe, Teff)

def lambda_pbh(M, z, xe, Teff):
    beta, gamma = beta_pbh(M, z, xe, Teff), gamma_pbh(M, z, xe, Teff)
    lam_ricotti = np.exp(4.5 / (3.0 + beta**0.75)) / (np.sqrt(1.0 + beta) + 1.0) ** 2
    lam_nodrag = 0.15 + (np.exp(1.5) / 4.0 - 0.15) * (gamma**2 / (88.0 + gamma**2)) ** 0.22
    return lam_ricotti * lam_nodrag / (np.exp(1.5) / 4.0)

def Mdot_pbh(M, z, xe, Teff):
    vB = 9.09e3 * np.sqrt((1.0 + xe) * Teff)
    return 9.15e22 * M**2 * ((1.0 + z) / vB) ** 3 * lambda_pbh(M, z, xe, Teff)

def TS_over_me_pbh(M, z, xe, Teff, collisional):
    tau = 1.5 / (5.0 + gamma_pbh(M, z, xe, Teff) ** (2.0 / 3.0))
    YS = (2.0 / (1.0 + xe)) * (tau / 4.0) * (1.0 - 2.5 * tau) ** (1.0 / 3.0) * 1836.0
    if collisional: YS *= ((1.0 + xe) / 2.0) ** 8
    return YS / (1.0 + YS / 0.27) ** (1.0 / 3.0)

def eps_over_mdot_pbh(M, z, xe, Teff, collisional):
    X = TS_over_me_pbh(M, z, xe, Teff, collisional)
    Gff = np.empty_like(X)
    low = X < 1.0
    Gff[low] = (4.0 / np.pi) * np.sqrt(2.0 / (np.pi * X[low])) * (1.0 + 5.5 * X[low] ** 1.25)
    Gff[~low] = (13.5 / np.pi) * (np.log(2.0 * X[~low] * 0.56146 + 0.08) + 4.0 / 3.0)
    return X / 1836.0 / 137.0 * Gff

def L_pbh(M, z, xe, Teff, collisional):
    Mdot = Mdot_pbh(M, z, xe, Teff)
    return (Mdot / (1.4e17 * M)) * eps_over_mdot_pbh(M, z, xe, Teff, collisional) * Mdot * 9.0e20

def L_Edd(M): return 1.26e38 * M

def c_code_prefactor(xe, Teff):
    return (np.sqrt(Teff / 1.21e-8) / 3.0e10) * 0.067 * (xe / (1.0 + xe)) * (1.1e13 / Teff)

def compute_feedback(M, z, xe, Teff, lum_ratio=None, collisional=False):
    if lum_ratio is None:
        lum_ratio = L_pbh(M, z, xe, Teff, collisional) / L_Edd(M)
    return lum_ratio * c_code_prefactor(xe, Teff) * (1.0 + gamma_pbh(M, z, xe, Teff) ** (1.0 / 3.0))

masses = [1.0, 1.0e2, 1.0e4]
column_map = {(1.0, True): 1, (1.0, False): 2, (1.0e2, True): 3, (1.0e2, False): 4, (1.0e4, True): 5, (1.0e4, False): 6}
mass_colors = {1.0: "#d62728", 1.0e2: "#5b1a69", 1.0e4: "#1f3fd4"}
mass_labels = {1.0: r"$1\,M_\odot$", 1.0e2: r"$10^2\,M_\odot$", 1.0e4: r"$10^4\,M_\odot$"}

print("\nRECONSTRUCTION CHECK AGAINST T_feedback.dat")
for M in masses:
    for collisional in (True, False):
        col = column_map[(M, collisional)]
        reconstructed = compute_feedback(M, z, xe, Teff_original, lum_ratio=luminosity_data[:, col])
        published = feedback_data[:, col]
        rel_err = np.max(np.abs(reconstructed - published) / np.maximum(np.abs(published), 1e-300))
        branch = "collisional" if collisional else "photoionization"
        print(f"M={M:g} M_sun, {branch:16s}: max relative diff = {rel_err:.3e}")

fig, ax = plt.subplots(figsize=(8.2, 5.9))

for M in masses:
    color = mass_colors[M]
    for collisional, ls in ((True, "-"), (False, "--")):
        col = column_map[(M, collisional)]
        published = feedback_data[:, col]
        zero_v = compute_feedback(M, z, xe, Tgas, collisional=collisional)

        ax.loglog(z, published, color=color, ls=ls, lw=2.4)
        ax.loglog(z, zero_v, color=color, ls=ls, lw=1.25, marker="o", ms=3.0, markevery=65, mfc="white", mew=0.9)

ax.axhline(1.0, color="black", lw=1.0, ls=":")
ax.set(xlim=(3.0e2, 2.0e4), ylim=(1.0e-8, 1.0e2), xlabel=r"$z$",
       ylabel=r"$\max\!\left(\dot T_{\mathrm{Compt},L}/\dot T\right)$",
       title=r"Figure 9 reconstruction and comparison with $v_{\rm rel}=0$")
ax.grid(True, which="both", alpha=0.20)

mass_handles = [Line2D([0], [0], color=mass_colors[M], lw=2.5, label=mass_labels[M]) for M in masses]
style_handles = [
    Line2D([0], [0], color="black", ls="-", lw=2.2, label="Published, collisional"),
    Line2D([0], [0], color="black", ls="--", lw=2.2, label="Published, photoionization"),
    Line2D([0], [0], color="black", ls="-", marker="o", mfc="white", ms=4, lw=1.2, label=r"$v_{\rm rel}=0$, collisional"),
    Line2D([0], [0], color="black", ls="--", marker="o", mfc="white", ms=4, lw=1.2, label=r"$v_{\rm rel}=0$, photoionization"),
]

mass_legend = ax.legend(handles=mass_handles, loc="lower right", frameon=False, title="PBH mass")
ax.add_artist(mass_legend)
ax.legend(handles=style_handles, loc="upper right", frameon=False, fontsize=8)

fig.tight_layout()
plt.show()
