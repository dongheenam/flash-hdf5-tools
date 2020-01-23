import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

from astropy.constants import G # gravitational constant

c_s = 0.2 * u.km / u.s
resolution = 512
M_tot = 1.5* 2 * 387.5 * u.solMass
L = 2 * u.pc
mach = 5

# box sizes
x_max = L.to(u.cm) / 2
print(f"xyz, max                : {x_max:.7E}")

# total mass
M_tot = M_tot.to(u.g)
print(f"total_mass              : {M_tot:.7E}")

# rho_ambient
rho_0 = (M_tot/L**3).to(u.g / u.cm**3)
print(f"rho_ambient             : {rho_0:.7E}")

# target velocity dispersion
sigma_v = (mach * c_s).to(u.cm / u.s)
print(f"velocity displersion    : {sigma_v:.7E}")
# turnover time
T_turb = (L/(2*sigma_v)).to(u.s)
print(f"turbulent turnover time : {T_turb:.7E}")

# sink accretion radius
x_min = (L / resolution).to(u.cm)
r_acc = 2.5 * x_min
print(f"sink_accretion_radius   : {r_acc:.7E}")

# sink density threshold
rho_th = np.pi * c_s**2 / ( 4*G*r_acc**2 )
rho_th = rho_th.to(u.g / u.cm**3)
print(f"sink_density_thres      : {rho_th:.7E}")
print(f"isothermal_dens_thres   : {rho_th/2:.7E}")

# virial parameter
sigma_v = c_s * mach
alpha_vir = (5*sigma_v**2*x_max) / (3*G*M_tot)
alpha_vir = alpha_vir.to(u.dimensionless_unscaled)
print(f"virial parameter        : {alpha_vir:.4f}")

# free-fall time
t_ff = np.sqrt(3*np.pi/(32*G*rho_0)).to(u.s)
print(f"free-fall time          : {t_ff:.4E}")
print(f"(in Turb. crs. time)    : {t_ff/T_turb:.4f}")
