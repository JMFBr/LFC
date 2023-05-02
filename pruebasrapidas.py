import numpy as np

def LFC(n_0,n_s0,n_c):
    # -- Computes 1 LFC given Nc, No and Nso

    n_0 = int(n_0)
    n_s0 = int(n_s0)
    n_c = int(n_c)

    L = np.array([[n_0, 0], [n_c, n_s0]])
    C = np.zeros((n_0, n_s0, 2)) # % Plane x Sat x Omega&M

    for i in range(1, n_0+1):   # Loop 1:N_0, si no especificas range inicial, el loop empieza en i=0
        for j in range(1, n_s0+1):
            B = 2*np.pi*np.array([[i - 1], [j - 1]])
            C[i-1,j-1,:] = np.transpose(np.linalg.solve(L, B))

    return C


## DATA
N_0 = 4 # Num planes
N_s0 = 11 # Num satellites/planes
N_c = 4 # Phasing parameter

mu = 3.986e14 # [m3/s2], Earth standard gravitational parameter
RE = 6371e3   # [m], Earth Radius
h = 580e3    # [m], Altitude

a = RE + h
e = 0
i = 72*np.pi/180 # [rad], Inclination
om = 0*np.pi/180  # [rad], argument of the perigee

C = LFC(N_0, N_s0, N_c)

Omega = C[:, :, 0] # RAAN matrix
M = C[:, :, 1] # Mean anomaly matrix

M_bool = np.logical_and(M >= 0, M <= 2*np.pi) # Check if values are between 0 and 2*pi
M[~M_bool] += 2*np.pi # Set negative values of M to M+2*pi


## MINIMUM DISTANCE

#  rho_min = The closest approach between the two satellites in two circular orbits
#
#  DO = DeltaOmega, Delta RAAN bw circular orbits
#  DM = DeltaMeanAnomaly, Delta mean anomaly bw satellites
#
#  sufficient to evaluate the minimum distance between the first satellite,
#  with all the other satellites staying on different orbital planes

# Initialize rho_min matrix

rho_min = np.zeros(M.shape)

for m in range(M.shape[0]):
    for n in range(M.shape[1]):
        DM = M[m, n] - M[0, 0]  # [rad]. Take first satellite as reference
        DO = Omega[m, n] - Omega[0, 0]

        DF = DM - 2 * np.arctan(-np.cos(i) * np.tan(DO/2))

        rho_min[m, n] = 2 * np.abs(np.sqrt(1 + np.cos(i)**2 + np.sin(i)**2 - np.cos(DO))/2) * np.sin(DF/2)  # [rad]

d_min = rho_min * (RE + h)  # [m]
d_min_km = d_min / 1000

# Find the minimum distance among all the satellite pairs (Different to 0)
non_zero_abs_vals = np.abs(d_min[d_min != 0])
if len(non_zero_abs_vals) == 0:
    min_distance = None
else:
    min_distance = np.min(non_zero_abs_vals) # [m]

print(min_distance/1000)









