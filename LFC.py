import numpy as np

#N_TS = 44 # Total number of satellites

N_0 = 4 # Number of planes
N_s0 = 11 # Number of sats/plane
N_c = 2 # Phasing parameter

L = np.array([[N_0, 0], [N_c, N_s0]])
C = np.zeros((N_0, N_s0, 2)) # % Plane x Sat x Omega&M

for i in range(1, N_0+1):
    for j in range(1, N_s0+1):
        B = 2*np.pi*np.array([[i - 1], [j - 1]])
        C[i-1,j-1,:] = np.transpose(np.linalg.solve(L, B))
