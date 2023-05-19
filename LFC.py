import numpy as np
from numpy import linalg as LA

## PRUEBA
N_TS = 44  # Total number of satellites

# Data
mu = 3.986e14  # [m3/s2], Earth standard gravitational parameter
RE = 6371e3  # [m], Earth Radius
h = 580e3  # [m], Altitude

a = RE + h  # [m], Semi-major axis
e = 0
inc = 72*np.pi/180  # [rad], Inclination
om = 0*np.pi/180  # [rad], Argument of the perigee

twin_d = 2*60  # [s], Twin fixed separation distance WAC-NAC
twin_d = twin_d*np.sqrt(mu/a**3)*(RE+h)  # [m]


def MinDist(Omega, M):
    # -- Compute rho_min = The closest approach between the two satellites in two circular orbits
    # -- Distance bw satellites in different orbital planes
    # INPUTS: Matrices for M & Omega from LFC method

    #  DO = DeltaOmega, Delta RAAN bw circular orbits
    #  DM = DeltaMeanAnomaly, Delta mean anomaly bw satellites

    #  It is sufficient to evaluate the minimum distance between the first satellite,
    #  with all the other satellites staying on different orbital planes (bc regular distribution w/ LFC)

    # Initialize rho_min matrix
    rho_min = np.zeros(M.shape)

    for m in range(M.shape[0]):
        for n in range(M.shape[1]):
            DM = M[m, n] - M[0, 0]  # [rad]. Take first satellite as reference
            DO = Omega[m, n] - Omega[0, 0]

            DF = DM - 2 * np.arctan(-np.cos(inc) * np.tan(DO / 2))

            rho_min[m, n] = 2 * np.abs(np.sqrt(1 + np.cos(inc) ** 2 + np.sin(inc) ** 2 - np.cos(DO)) / 2) * np.sin(
                DF / 2)  # [rad]

    d_min = rho_min * (RE + h)  # [m]
    d_min_km = d_min / 1000

    # Find the minimum distance among all the satellite pairs (Different to 0)
    non_zero_abs_vals = np.abs(d_min[d_min != 0])
    if len(non_zero_abs_vals) == 0:
        min_distance = None
    else:
        min_distance = np.min(non_zero_abs_vals)  # [m]

    # print(min_distance / 1000)
    return min_distance


def MaxDist():
    # -- Maximum allowable distance among satellites in same plane
    # -- Set by ISL constraint, satellites must see each other

    # theta = angle bw 2 satellites
    # alpha = Maximum angle at which 2 satellites see each other, determined by taking into account atmospheric
    # effects at h=100km

    h_atm = 100e3  # [m], altitude at which we take into account atmospheric effects

    alpha = np.arccos((RE+h_atm)/(RE+h))  # [rad], 2 sates see each other if theta<=2*alpha

    max_dist = 2*alpha*(RE+h)

    return max_dist


def LFC(n_0, n_s0, n_c):
    # -- Computes 1 LFC given Nc, No and Nso

    n_0 = int(n_0)
    n_s0 = int(n_s0)
    n_c = int(n_c)

    L = np.array([[n_0, 0], [n_c, n_s0]])
    C = np.zeros((n_0, n_s0, 2))  # % Plane x Sat x Omega&M

    Omega = np.ones((N_TS, 1))
    M = np.ones((N_TS, 1))
    k = 0

    for i in range(1, n_0 + 1):  # Loop 1:N_0, si no especificas range inicial, el loop empieza en i=0
        for j in range(1, n_s0 + 1):
            B = 2 * np.pi * np.array([[i - 1], [j - 1]])
            C[i - 1, j - 1, :] = np.transpose(np.linalg.solve(L, B))

            Omega[k, :] = C[i - 1, j - 1, 0]  # RAAN in vector form
            M[k, :] = C[i - 1, j - 1, 1]  # Mean anomaly in vector form
            k = k + 1

    Omega_m = C[:, :, 0]  # RAAN matrix
    M_m = C[:, :, 1]  # Mean anomaly matrix

    M_bool = np.logical_and(M >= 0, M <= 2 * np.pi)  # Check if values are between 0 and 2*pi
    M[~M_bool] += 2 * np.pi  # Set negative values of M to M+2*pi

    M_bool = np.logical_and(M_m >= 0, M_m <= 2 * np.pi)  # Same for the matrix
    M_m[~M_bool] += 2 * np.pi

    return C, Omega, M, Omega_m, M_m


def NumSats(n_TS):
    # -- Given Total number of satellites N_TS, compute all [N0 Ns0] possible pairs

    # Create an array of integers from 1 to N_TS
    all_integers = np.arange(1, n_TS + 1)
    # Find the divisors of N_TS using boolean indexing
    divisors = all_integers[n_TS % all_integers == 0]

    n_0 = np.zeros(divisors.shape)
    n_0[:] = divisors # Number of planes, dim(1x#multiples)

    n_s0 = n_TS/n_0  # Number of sats/plane, dim(1x#multiples)

    return n_s0, n_0


def solidAngle(h0, SW):
    # Compute solid angle of the sensor given the swath width
    # -- h0 = [m], Altitude used in the sensor information, 500km for Simera
    # -- SW = [m], Sensor Swath Width, 120km at 500km for Simera

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

    return alpha


def ConstFam(n_TS):
    # -- 1. Loop all combination pairs n_0&n_s0
    # -- 2. Loop all possible n_c for each pair
    # For each constellation, inside the 2nd loop compute minimum distance constraint and coverage

    n_0, n_s0 = NumSats(n_TS)

    for j in range(len(n_0)):
        n_c = np.arange(1, n_0[j]) # Nc is in the range [1, N0-1]

        for k in range(len(n_c)):
            C, Omega, M, Omega_m, M_m = LFC(n_0[j], n_s0[j], n_c[k])
            # CONSTELLATION OBTAINED, HERE COMPUTE COVERAGE AND DISTANCE CONSTRAINTS

            # MIN Distance constraint:
            min_dist = MinDist(Omega, M)  # [m], Min distance inter-planes
            if min_dist < 2*twin_d:
                # Discard constellation if distance requirements are not met
                continue

            # MAX Distance constraint:
            WAC_dist = 2*np.pi/n_s0[j]*(RE+h)  # [m], WAC-WAC satellites distance in 1 plane
            NAC_dist = twin_d  # [m], WAC-NAC distance in 1 plane
            max_dist = MaxDist()  # [m], Max distance ISL constraint within 1 plane

            if WAC_dist > (NAC_dist + max_dist):
                # Discard constellation if ISL cannot be connected (WAC1-NAC1--WAC2)
                continue

            # Create constellation matrix
            const_matrix = np.ones((N_TS, 4))
            const_matrix[:, 0] = a  # [m]
            const_matrix[:, 1] = e
            const_matrix[:, 2] = inc  # [rad]
            const_matrix[:, 3] = om  # [rad]
            const_matrix = np.c_[const_matrix, Omega, M]  # Constellation matrix: (Nts x 6 OEs)

            # Transform constellation matrix: OEs to ECI (Nts x 6)

            # Transform constellation matrix: ECI to ECEF (Nts x 6)

            # Read target list

            # Transform target matrix: LatLon to ECEF


            ## COVERAGE AND TARGET ACCESS

            # Compute timestep

            # Create coverage matrix: (Num targets x TimeStep)

            # Transform target matrix: ECEF to UrUhUy















