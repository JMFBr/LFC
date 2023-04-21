import numpy as np

## PRUEBA
# #N_TS = 44 # Total number of satellites
#
# N_0 = 4 # Number of planes
# N_s0 = 11 # Number of sats/plane
# N_c = 2 # Phasing parameter

def LFC(n_0,n_s0,n_c):
    # -- Computes 1 LFC given Nc, No and Nso

    L = np.array([[n_0, 0], [n_c, n_s0]])
    C = np.zeros((n_0, n_s0, 2)) # % Plane x Sat x Omega&M

    for i in range(1, n_0+1):   # Loop 1:N_0, si no especificas range inicial, el loop empieza en i=0
        for j in range(1, n_s0+1):
            B = 2*np.pi*np.array([[i - 1], [j - 1]])
            C[i-1,j-1,:] = np.transpose(np.linalg.solve(L, B))

    return C


def NumSats(n_TS):
    # -- Given Total number of satellites N_TS, compute N0, Ns0 and Nc

    n_0 = [] # Number of planes, Initialize n_0

    for i in range(1, n_TS+1): # Find multiples of n_TS
        if n_TS % i == 0:
            n_0.append(i)


    n_s0 = n_TS/n_0 # Number of sats/plane

    for j in range(len(n_0)):
        n_c = np.arange(1, n_0[j] + 1)

        for k in range(len(n_c)):
            C = LFC(n_0[j], n_s0[j], n_c[k])

    return C










