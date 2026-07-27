from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# physical constants 
C_LIGHT = 2.99792458e10
K_B = 1.380649e-16
M_P = 1.67262192369e-24
MP_OVER_ME = 1836.0
T0 = 2.7255

# load the HyRec recombination history
HYREC_OUTPUT = Path(__file__).resolve().parent / "HYREC-2-master/HYREC-2-master/output_xe.dat"

def load_hyrec(path, z_min=200.0, z_max=2.0e4):
    data = np.loadtxt(path)
    z, xe, tgas = data[:, 0], data[:, 1], data[:, 2]
    mask = np.isfinite(z) & np.isfinite(xe) & np.isfinite(tgas) & (z >= z_min) & (z <= z_max) & (tgas > 0.0)
    order = np.argsort(z[mask])
    return z[mask][order], xe[mask][order], tgas[mask][order]

def vbc_rms(z):
    return np.minimum(1.0, z / 1.0e3) * 30.0e5

def velocity_grid(n=500, xmax=5.0):
    x = np.linspace(0.0, xmax, n)
    weight = x**2 * np.exp(-1.5 * x**2)
    return x, weight / np.trapezoid(weight, x)

def teff_from_velocity(tgas, xe, vrel):
    return tgas + (M_P / K_B) * vrel**2 / (1.0 + xe)

def bondi_speed(xe, teff):
    return 9.09e3 * np.sqrt((1.0 + xe) * teff)

# bondi accretion model
def beta_pbh(mass, z, xe, teff):
    vb = bondi_speed(xe, teff)
    return 7.45e-24 * xe * (1.33e26 * mass / vb**3) / (1.0 / (1.0 + z))**4

def gamma_pbh(mass, z, xe, teff):
    return 2.0 * MP_OVER_ME / (1.0 + xe) * beta_pbh(mass, z, xe, teff)

def lambda_pbh(mass, z, xe, teff):
    beta = beta_pbh(mass, z, xe, teff)
    gamma = gamma_pbh(mass, z, xe, teff)
    lam_ad, lam_iso = 0.25 * 0.6**1.5, 0.25 * np.exp(1.5)
    
    lam_drag = np.exp(4.5 / (3.0 + beta**0.75)) / (np.sqrt(1.0 + beta) + 1.0)**2
    lam_nodrag = lam_ad + (lam_iso - lam_ad) * (gamma**2 / (88.0 + gamma**2))**0.22
    return lam_drag * lam_nodrag / lam_iso

def mdot_pbh(mass, z, xe, teff):
    vb = bondi_speed(xe, teff)
    return 9.15e22 * mass**2 * ((1.0 + z) / vb)**3 * lambda_pbh(mass, z, xe, teff)

def ts_over_me(mass, z, xe, teff, branch):
    gamma = gamma_pbh(mass, z, xe, teff)
    tau = 1.5 / (5.0 + gamma**(2.0 / 3.0))
    
    if branch == "collisional":
        omega = np.sqrt(np.maximum(2.0 - 5.0 * tau, 0.0))
        y = ((1.0 + xe) / 2.0)**7 * (tau / 2.0) * (omega / 4.0)**(2.0 / 3.0) * MP_OVER_ME
    elif branch == "photoionization":
        y = (2.0 / (1.0 + xe)) * (tau / 4.0) * (1.0 - 2.5 * tau)**(1.0 / 3.0) * MP_OVER_ME
    else:
        raise ValueError("branch must be 'collisional' or 'photoionization'")
        
    y = np.maximum(y, 1.0e-300)
    return y / (1.0 + y / 0.27)**(1.0 / 3.0)

def free_free_j(x):
    x = np.maximum(np.asarray(x), 1.0e-300)
    result = np.empty_like(x)
    low = x < 1.0
    result[low] = (4.0 / np.pi) * np.sqrt(2.0 / (np.pi * x[low])) * (1.0 + 5.5 * x[low]**1.25)
    result[~low] = (27.0 / (2.0 * np.pi)) * (np.log(2.0 * x[~low] * 0.56146 + 0.08) + 4.0 / 3.0)
    return result

def luminosity(mass, z, xe, teff, branch):
    mdot = mdot_pbh(mass, z, xe, teff)
    eps_over_mdot = ts_over_me(mass, z, xe, teff, branch) / MP_OVER_ME / 137.0 * free_free_j(ts_over_me(mass, z, xe, teff, branch))
    return (mdot / (1.4e17 * mass)) * eps_over_mdot * mdot * 9.0e20

def l_edd(mass):
    return 1.4e17 * mass * 9.0e20

# thermal-feedback ratio (eq. 66)
def feedback_ratio(mass, z, xe, teff, branch):
    vb = bondi_speed(xe, teff)
    gamma = gamma_pbh(mass, z, xe, teff)
    lum = luminosity(mass, z, xe, teff, branch)
    tcmb = T0 * (1.0 + z)
    return (0.07 * (xe / (1.0 + xe)) * (lum / l_edd(mass)) * (vb / C_LIGHT) 
            * (M_P * C_LIGHT**2 / (K_B * tcmb)) * np.sqrt(1.0 + gamma**(2.0 / 3.0)))

# average over the PBH velocity distribution
def average_feedback(mass, z, xe, tgas, branch):
    x, weight = velocity_grid()
    vrel = x[:, None] * vbc_rms(z)[None, :]
    teff = teff_from_velocity(tgas[None, :], xe[None, :], vrel)
    feedback = feedback_ratio(mass, z[None, :], xe[None, :], teff, branch)
    return np.trapezoid(weight[:, None] * feedback, x, axis=0)

# plot figure 9
def main():
    z, xe, tgas = load_hyrec(HYREC_OUTPUT)
    masses, colors = [1.0, 1.0e2, 1.0e4], ["red", "purple", "blue"]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    plotted_values = []

    for mass, color in zip(masses, colors):
        coll = average_feedback(mass, z, xe, tgas, "collisional")
        photo = average_feedback(mass, z, xe, tgas, "photoionization")
        plotted_values.extend([coll, photo])

        label = r"$1\,M_\odot$" if mass == 1.0 else rf"$10^{{{int(np.log10(mass))}}}\,M_\odot$"
        ax.loglog(z, coll, color=color, lw=2.3, ls="-", label=label)
        ax.loglog(z, photo, color=color, lw=2.3, ls="--")

 # figure formatting
    all_vals = np.concatenate(plotted_values)
    pos_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0.0)]
    
    ax.set(xlim=(2.0e2, 2.0e4), ylim=(max(1.0e-10, pos_vals.min() / 2.0), pos_vals.max() * 2.0),
           xlabel=r"Redshift, $z$", ylabel=r"$\langle \max(\dot{T}_{\rm Compt,L}/\dot{T}) \rangle$",
           title="Thermal-feedback estimate from Eq. (66)")
    
    ax.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.8)
    ax.grid(True, which="major", lw=0.7, alpha=0.30)
    ax.grid(True, which="minor", lw=0.4, alpha=0.12)
    ax.tick_params(which="both", direction="in", top=True, right=True)

# legend
    mass_legend = ax.legend(title="PBH mass", loc="lower right", frameon=False, fontsize=9)
    ax.add_artist(mass_legend)
    
    branch_handles = [
        plt.Line2D([0], [0], color="black", lw=2.2, ls="-", label="Collisional ionization"),
        plt.Line2D([0], [0], color="black", lw=2.2, ls="--", label="Photoionization"),
    ]
    ax.legend(handles=branch_handles, loc="upper right", bbox_to_anchor=(1.0, 0.94), frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / "figure9_equation66.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
