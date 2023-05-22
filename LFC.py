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

def kep2eci(const_m_OE, t, T):  # R modified for matrices
    """
    Converts Keplerian orbital elements to Earth-centered inertial coordinates.
    Parameters:
    -----------
    const_m_OE : constellation matrix with keplerian elements, array_like (N_TS x 6)
        - ROWS: Orbital elements in the following order: semi-major axis (a) [m], eccentricity (e),
        inclination (i) [rad], argument of peri-apsis (omega) [rad], right ascension of ascending node (Omega) [rad],
        mean anomaly (M) [rad].
        - COLUMNS: Satellites in the constellation

    Returns:
    --------
    const_m_ECI : constellation matrix in ECI coordinates, array_like (N_TS x 6)
        - ROWS: x_eci (x3), Earth-centered inertial coordinates in meters: x, y, z (m).
                v_eci (x3), Earth-centered inertial velocity coordinates: vx, vy, vz (m/s).
        - COLUMNS: Satellites in the constellation
    """
    # Define Keplerian orbital elements
    a_v = const_m_OE[:, 0]
    e_v = const_m_OE[:, 1]
    i_v = const_m_OE[:, 2]
    omega_v = const_m_OE[:, 3]
    Omega_v = const_m_OE[:, 4]

    # Initial position
    M0_v = const_m_OE[:, 5]  # Initial mean anomaly vector
    t0_v = M0_v*T/(2*np.pi)  # [s], Time from peri-apsis that corresponds to the initial true anomaly vector

    # Current position
    M_v = 2 * np.pi * (t0_v + t) / T  # Mean anomaly at current position

    # Calculate eccentric anomaly
    E = M_v
    nu = np.zeros(N_TS)
    for j in range(len(E)):  # Compute E using Newton's method for each satellite in teh constellation
        while True:
            E_new = E[j] + (M_v[j] - E[j] + e_v[j] * np.sin(E[j])) / (1 - e_v[j] * np.cos(E[j]))
            if abs(E_new - E[j]) < 1e-8:
                break
            E[j] = E_new

        if E[j] < 0:
            E[j] += 2 * np.pi

        # Calculate true anomaly corresponding to the current time t
        nu[j] = 2 * np.arctan(np.sqrt((1 + e_v[j]) / (1 - e_v[j])) * np.tan(E[j] / 2))
        if nu[j] < 0:
            nu[j] += 2 * np.pi

    # Calculate semi-latus rectum and mean motion
    p = a_v * (1 - e_v ** 2)
    # Calculate distance from Earth to satellite
    r = p / (1 + e_v * np.cos(nu))

    # Calculate position and velocity in peri-focal coordinates

    x = np.array([r * np.cos(nu)])
    y = np.array([r * np.sin(nu)])
    z = np.zeros((N_TS, 1))
    r_pqw = np.concatenate((x.T, y.T, z), axis=1)  # [m], Matrix: (Position vectors in peri-focal x N_TS)

    vx = np.array([-np.sin(nu)]) * np.sqrt(mu / p)
    vy = np.array([e + np.cos(nu)]) * np.sqrt(mu / p)
    vz = np.zeros((N_TS, 1))
    v_pqw = np.concatenate((vx.T, vy.T, vz), axis=1)  # [m/s], Matrix: (Velocity vectors in peri-focal x N_TS)

    # Transformation matrix from peri-focal to geocentric equatorial coordinates. Dimensions: (3 x 3 x N_TS)
    R_pqw_to_eci = np.array([
        [np.cos(Omega_v) * np.cos(omega_v) - np.sin(Omega_v) * np.sin(omega_v) * np.cos(i_v),
         -np.cos(Omega_v) * np.sin(omega_v) - np.sin(Omega_v) * np.cos(omega_v) * np.cos(i_v), np.sin(Omega_v) * np.sin(i_v)],
        [np.sin(Omega_v) * np.cos(omega_v) + np.cos(Omega_v) * np.sin(omega_v) * np.cos(i_v),
         -np.sin(Omega_v) * np.sin(omega_v) + np.cos(Omega_v) * np.cos(omega_v) * np.cos(i_v), -np.cos(Omega_v) * np.sin(i_v)],
        [np.sin(i_v) * np.sin(omega_v), np.sin(i_v) * np.cos(omega_v), np.cos(i_v)]
    ])

    # Convert
    r_eci = np.zeros((N_TS, 3))
    v_eci = np.zeros((N_TS, 3))
    for j in range(N_TS):
        r_eci[j, :] = np.dot(R_pqw_to_eci[:, :, j], r_pqw[j, :])
        v_eci[j, :] = np.dot(R_pqw_to_eci[:, :, j], v_pqw[j, :])

    const_m_ECI = np.concatenate((r_eci, v_eci), axis=1)

    return const_m_ECI


def read_targets():
    """Choose a season to get the targets
        Summer for now
    """

    target_m = pd.read_csv("summer.csv")
    # target_m = np.loadtxt("summer.csv", delimiter=",", dtype=str)  # Target matrix: Lat - Lon - Weight

    target_m = target_m.to_numpy()  # Target matrix: Lat - Lon - Weight

    target_m[:, 0] = np.radians(target_m[:, 0])  # Latitude to radians
    target_m[:, 1] = np.radians(target_m[:, 1])  # Longitude to radians

    weight = target_m[:, 2]

    return target_m, weight


def latlon2car(target_m):
    """ Transform coordinates from Lat-Lon to ECEF:
        Lat = target_m[:, 0]
        Lon = target_m[:, 1]
        Weight = target_m[:, 2]
    """
    x = RE * np.cos(target_m[:, 0]) * np.cos(target_m[:, 1])
    y = RE * np.cos(target_m[:, 0]) * np.sin(target_m[:, 1])
    z = RE * np.sin(target_m[:, 0])

    target_m_r = np.array([x, y, z])

    print('Targets position vectors calculated \n')
    return target_m_r


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

            # Create constellation matrix with all satellites' orbital elements
            const_OE = np.ones((N_TS, 4))
            const_OE[:, 0] = a  # [m]
            const_OE[:, 1] = e
            const_OE[:, 2] = inc  # [rad]
            const_OE[:, 3] = om  # [rad]
            const_OE = np.c_[const_OE, Omega, M]  # Constellation matrix: (Nts x 6 OEs)

            # Transform constellation matrix: OEs to ECI (Nts x 6)
            t = 0
            T = 2 * np.pi * np.sqrt(a ** 3 / mu)  # [s], Orbital period
            const_ECI = kep2eci(const_OE, t, T)

            # Transform constellation matrix: ECI to ECEF (Nts x 6)

            # Read target list
            target_m_LatLon, weight = read_targets()  # Target matrix: Lat - Lon, Weight

            # Transform target matrix: LatLon to ECEF
            target_m_ECEF = latlon2car(target_m_LatLon)  # Target matrix in ECEF: x - y -z


            ## COVERAGE AND TARGET ACCESS

            # Compute timestep

            # Create coverage matrix: (Num targets x TimeStep)

            # Transform target matrix: ECEF to UrUhUy















