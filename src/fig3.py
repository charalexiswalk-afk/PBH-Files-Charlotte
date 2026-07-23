import numpy as np
import matplotlib.pyplot as plt

lambda_ad = 0.25 * (3.0 / 5.0)**1.5
lambda_iso = 0.25 * np.exp(1.5)

gamma = np.logspace(-3, 3, 500)

lam = lambda_ad + (lambda_iso - lambda_ad) * ((gamma**2) / (88.0 + gamma**2))**0.22

plt.figure(figsize=(6, 4.5))

plt.semilogx(gamma, lam, color="purple", linewidth=2.2, label="Analytic fit")
plt.axhline(lambda_ad, color="black", linestyle=":", linewidth=1.5)
plt.axhline(lambda_iso, color="black", linestyle=":", linewidth=1.5)

plt.text(3e-3, lambda_ad + 0.08,
         r"$\lambda_{\rm ad}\approx 0.12$", fontsize=10)

plt.text(3e-3, lambda_iso - 0.08,
         r"$\lambda_{\rm iso}\approx 1.12$", fontsize=10)

plt.xlabel(r"$\gamma$", fontsize=12)
plt.ylabel(r"$\lambda$", fontsize=12)
plt.title(r"Dimensionless Accretion Rate vs. Compton Cooling", fontsize=12)

plt.xlim(1e-3, 1e3)
plt.ylim(0, 1.2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("lambda_ad =", lambda_ad)
print("lambda_iso =", lambda_iso)
