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


## TARGET ACCESS

h0 = 500e3  # Altitude used in the sensor information
SW = 120e3
psi = SW / (2 * RE)

# Using Newtons method, solve: psi = -eps + acos(RE/(RE + h0)*cos(eps)) for elevation angle given psi, RE and h
err = 1e-8  # Error
eps = 1 * np.pi / 180  # [rad], Initial value
div = 1

while np.abs(div) > err:
    f = -eps + np.arccos(RE / (RE + h0) * np.cos(eps)) - psi  # Equation to solve = 0
    df = -1 + 1 / (np.sqrt(1 - (RE / (RE + h0) * np.cos(eps)) ** 2)) * RE / (RE + h0) * np.sin(
        eps)  # Derivative of equation

    div = f / df
    eps = eps - div

alpha = np.pi / 2 - psi - eps  # [rad], solid angle






