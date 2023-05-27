import numpy as np
import pandas as pd
from numpy import linalg as LA

## PRUEBA
N_TS = 44  # Total number of satellites

# Orbit Data
mu = 3.986e14  # [m3/s2], Earth standard gravitational parameter
RE = 6371e3  # [m], Earth Radius
h = 580e3  # [m], Altitude
a = RE + h  # [m], Semi-major axis
e = 0
inc = 72 * np.pi/180  # [rad], Inclination
om = 0 * np.pi/180  # [rad], Argument of the perigee

# Twin data
twin_d = 2*60  # [s], Twin fixed separation distance WAC-NAC
twin_d = twin_d * np.sqrt(mu / a ** 3) * (RE + h)  # [m]

# Orbit propagation data
J2 = 0.00108263
n0 = np.sqrt(mu / a**3)  # Unperturbed mean motion
K = (RE / (a * (1 - e**2)))**2

# Sensor info (Simera)
h_s = 500e3  # [m], Altitude at which the sensor information is given
d_ac = 120e3  # [m], Swath width
d_al = 120e3  # [m], Along distance: used only for the simulation as the scanner is pushbroom

# Times
v_s = np.sqrt(mu/a)  # [m/s], Satellite velocity in a circular orbit
Dt = a / RE * d_al / v_s  # [s], Timestep
t_s = 24*3600  # Simulation duration
time_array_initial = (2023, 6, 26, 5, 43, 12)  # year, month, day, hour, minute, second (UTC)


# CONSTRAINTS
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


# CONSTELLATION
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


# REFERENCE FRAME CHANGES & TARGETS
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
    M0_v = const_m_OE[:, 5]  # Initial mean anomaly vector of values
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


def eci2ecef(time_array, const_m_ECI):
    """
    Converts ECI coordinates to Earth-centered, Earth-fixed coordinates.
    Parameters:
    -----------
    time_array: Start date and time
    const_m_ECI: Constellation matrix with ECI coordinates, array_like (N_TS x 6)
        - ROWS: Position vector (x3), Velocity vector (x3).
        - COLUMNS: Satellites in the constellation

    Returns:
    --------
    const_m_ECEF : constellation matrix in ECEF coordinates, array_like (N_TS x 6)
        - ROWS: x_ecef (x3), Earth-centered Earth fixed coordinates in meters: x, y, z (m).
                v_ecef (x3), Earth-centered Earth fixed velocity coordinates: vx, vy, vz (m/s).
        - COLUMNS: Satellites in the constellation
    """

    Y = time_array[0]  # year
    Mo = time_array[1]  # month
    D = time_array[2]  # day
    ho = time_array[3]  # hour (in UTC time)
    mi = time_array[4]  # minutes (in UTC time), adding the time since the start of the imaging
    se = time_array[5]  # seconds (in UTC time)

    jd = 1721013.5 + 367 * Y - int(7 / 4 * (Y + int((Mo + 9) / 12))) + int(275 * Mo / 9) + D + (
                60 * ho + mi) / 1440 + se / 86400

    # Calculate the number of days since J2000.0
    days_since_J2000 = jd - 2451545.0

    # Calculate the GMST in hours, minutes, and seconds
    GMST_hours = 18.697374558 + 24.06570982441908 * days_since_J2000

    GMST_hours %= 24  # Convert to the range [0, 24)
    gmst = 2 * np.pi * GMST_hours / 24

    R_ECEF2ECI = np.array([
        [np.cos(gmst), -np.sin(gmst), 0],
        [np.sin(gmst), np.cos(gmst), 0],
        [0, 0, 1],
    ])

    # Earth rotation rate.
    w = 7.292115146706979e-5

    # Rotation
    r_ecef = np.zeros((N_TS, 3))
    v_ecef = np.zeros((N_TS, 3))
    for k in range(N_TS):
        r_ecef[k, :] = np.dot(np.transpose(R_ECEF2ECI), np.array(const_m_ECI[k, 0:3]))
        v_ecef[k, :] = np.dot(np.transpose(R_ECEF2ECI), np.array(const_m_ECI[k, 3:])) - np.cross([0, 0, w], r_ecef[k, :])

    const_m_ECEF = np.concatenate((r_ecef, v_ecef), axis=1)

    return const_m_ECEF


def read_targets():
    """
    Import target list depending on the season
    """

    if time_array_initial[1] <= 3:  # Jan to March included: winter
        target_m = pd.read_csv("winter.csv")
        print('Winter')
    if 4 <= time_array_initial[1] <= 6:  # April to June: spring
        target_m = pd.read_csv("spring.csv")
        print('Spring')
    if 7 <= time_array_initial[1] <= 9:  # July to Sep: summer
        target_m = pd.read_csv("fall.csv")
        print('Summer')
    if time_array_initial[1] >= 10:  # Oct to Dec: fall
        target_m = pd.read_csv("fall.csv")
        print('Fall')

    target_m = target_m.to_numpy()  # Target matrix: Lat - Lon - Weight

    target_m[:, 0] = np.radians(target_m[:, 0])  # Latitude to radians
    target_m[:, 1] = np.radians(target_m[:, 1])  # Longitude to radians

    weight = target_m[:, 2]

    return target_m, weight


def latlon2ecef(target_m):
    """
    Transform coordinates from Lat-Lon to ECEF:
        Lat, [rad] = target_m[:, 0]
        Lon, [rad] = target_m[:, 1]
        Weight = target_m[:, 2]
    """
    x = RE * np.cos(target_m[:, 0]) * np.cos(target_m[:, 1])
    y = RE * np.cos(target_m[:, 0]) * np.sin(target_m[:, 1])
    z = RE * np.sin(target_m[:, 0])

    target_m_r = np.array([x, y, z])

    print('Targets position vectors calculated \n')
    return target_m_r


def latlon2ecef_elips(target_m):
    """
    Transform coordinates from Lat-Lon to ECEF:
        Lat, [rad] = target_m[:, 0]
        Lon, [rad] = target_m[:, 1]
        Weight = target_m[:, 2]
    """

    alt = 0  # [m], Altitude of targets (assumed 0 for now)

    # Define WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis (m)
    b = 6356752.0  # semi-minor axis (m)

    f = 1 - b / a  # flattening of Earth's ellipsoid
    e2 = 1 - b ** 2 / a ** 2  # square of the first numerical eccentricity of Earth's ellipsoid

    N = a / np.sqrt(1 - e2 * np.sin(target_m[:, 0]) ** 2)

    x = (N + alt) * np.cos(target_m[:, 0]) * np.cos(target_m[:, 1])
    y = (N + alt) * np.cos(target_m[:, 0]) * np.sin(target_m[:, 1])
    z = ((1 - f) ** 2 * N + alt) * np.sin(target_m[:, 0])

    target_m_r = np.array([x, y, z])
    target_m_r = np.transpose(target_m_r)

    print('Targets position vectors calculated \n')
    return target_m_r


# COVERAGE
def unit_v(v):  # D
    u_v = v / LA.norm(v, axis=0)  # direction cosine

    return u_v


def projections(const_m_ECEF, target_m_ECEF):  # D modified
    """
    Project Target coordinates into [ur, uh, uy] RF
    """
    r = const_m_ECEF[:, 0:3]  # (N_TS, 3)
    v = const_m_ECEF[:, 3:]  # (N_TS, 3)

    u_r = np.apply_along_axis(unit_v, 1, r)  # (N_TS, 3)
    u_v = np.apply_along_axis(unit_v, 1, v)  # (N_TS, 3)

    u_r_t = np.apply_along_axis(unit_v, 1, target_m_ECEF)  # (N_targets, 3), unit vector in target direction ECEF

    print('New unit vectors calculated')

    u_h = np.cross(u_r, u_v, axisa=1, axisb=1, axisc=1) # (N_TS, 3)
    u_y = np.cross(u_h, u_r, axisa=1, axisb=1, axisc=1) # (N_TS, 3)

    print('New system reference calculated')

    # Target projection on new system of reference

    p1 = np.dot(u_r, u_r_t.T)  # (N_targets, N_TS), cos(angle) bw u_r and target position vector in ECEF
    p2 = np.dot(u_y, u_r_t.T)  # (N_targets, N_TS), cos(angle) bw u_y and target position vector in ECEF
    p3 = np.dot(u_h, u_r_t.T)  # (N_targets, N_TS), cos(angle) bw u_h and target position vector in ECEF

    print('Targets projections calculated')

    return p1, p2, p3


def filt_steps_fun(const_m_ECEF, target_m_ECEF, a_alfa, a_beta):  # D modified

    p1, p2, p3 = projections(const_m_ECEF, target_m_ECEF)

    # If the cosine is negative, means the satellite is in the other side of the Earth thus not visible
    mask_p1 = p1 > 0  # Boolean, mask_p1(i)=True if p1(i)>0, p1=tr.ur must be >0 always

    # ACROSS
    filt_steps_ac = np.absolute(p3) / p1 <= np.tan(a_alfa)  # Boolean, True if tan(alpha_t)<=tan(alpha_s)
    filt_steps_ac[~mask_p1] = False  # Values in mask_p1 that correspond to False are set to False in filt_steps_ac

    print('Across filter ok ', np.sum(filt_steps_ac))

    # ALONG TRACK
    filt_steps_al = np.absolute(p2) / p1 <= np.tan(a_beta)
    filt_steps_al[~mask_p1] = False

    print('Along filter ok', np.sum(filt_steps_al))

    filt_steps = np.logical_or(filt_steps_al, filt_steps_ac)  # Account covered targets for along and across angles

    print('Total filter ok', np.sum(filt_steps))

    return filt_steps_ac


def filt_pop(const_m_ECEF, target_m_ECEF, a_alfa, a_beta):  # D modified

    filt_steps = filt_steps_fun(const_m_ECEF, target_m_ECEF, a_alfa, a_beta)  # Boolean, True if target is covered

    cov_stepss = np.array(np.nonzero(filt_steps[:]))
    # Row 1: The row indices of True values in filt_steps
    # Row 2: The column indices of True values in filt_steps

    return cov_stepss


# ORBIT PROPAGATION
def propagation(const_m_OE):
    """
    IN:
    :param const_m_OE: Constellation matrix with OEs of previous timestep (a, e, i, om, Om, M)

    OUT:
    :return: const_m_OE_new: Constellation matrix with new OEs
    """

    om_dot = 3/2 * J2 * K * n0 * (2 - 5/2*np.sin(inc)**2)  # Argument of the perigee change rate due to J2
    Om_dot = -3 / 2 * J2 * K * n0 * np.cos(inc)  # RAAN change rate due to J2
    th_dot = n0 * (1 + 3/4 * J2 * K * (2 - 3*np.sin(inc)**2) * np.sqrt(1 - e**2))  # True anomaly change rate due to J2

    # Change True anomaly to Eccentric anomaly to Mean anomaly:
    D_th = th_dot * Dt  # Delta true anomaly //  Dt: Time step, computed at the start of the code
    D_E = np.arcsin((np.sin(D_th) * np.sqrt(1 - e**2))/(1 + e*np.cos(D_th)))  # Delta eccentric anomaly
    D_M = D_E - e * np.sin(D_E)  # Delta eman anomaly

    const_m_OE_new = const_m_OE.copy()  # a, e, i: no change
    const_m_OE_new[:, 3] += Dt * om_dot  # New arg of perigee
    const_m_OE_new[:, 4] += Dt * Om_dot  # New RAAN
    const_m_OE_new[:, 5] += D_M  # New mean anomaly

    return const_m_OE_new


def ConstFam(n_TS):
    """
    -- 1. Loop all combination pairs n_0&n_s0
    -- 2. Loop all possible n_c for each pair
    For each constellation, inside the 2nd loop compute minimum distance constraint and coverage
    """
    n_0, n_s0 = NumSats(n_TS)

    for j in range(len(n_0)):
        n_c = np.arange(1, n_0[j]) # Nc is in the range [1, N0-1]

        for k in range(len(n_c)):
            # 1. CONSTELLATION
            C, Omega, M, Omega_m, M_m = LFC(n_0[j], n_s0[j], n_c[k])

            # 2. CONSTRAINTS
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

            # 3. CONSTELLATION MATRIX AND TRANSFORMATIONS
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
            const_ECEF = eci2ecef(time_array_initial, const_ECI)

            # 4. TARGET LIST
            # Read target list
            target_LatLon, weight = read_targets()  # Target matrix Lat - Lon (N_targets, 2); Weight (N_targets, 1)
            # Transform target matrix: LatLon to ECEF
            # target_m_ECEF = latlon2ecef(target_LatLon)  # Target matrix in ECEF (N_targets, 3): x - y -z
            target_ECEF = latlon2ecef_elips(target_LatLon)  # Target matrix in ECEF (N_targets, 3): x - y -z, Ellipsoid

            ## COVERAGE AND TARGET ACCESS

            # Compute timestep

            # Create coverage matrix: (Num targets x TimeStep)
            # Transform target matrix: ECEF to UrUhUy

            eta = a / RE
            f_acr = solidAngle(h_s, d_ac)  # [rad]
            f_alo = solidAngle(h_s, d_al)  # [rad]
            an_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))
            an_alfa = an_alfa.T

            an_beta = - f_alo + np.arcsin(eta * np.sin(f_alo))
            an_beta = an_beta.T

            cover = filt_steps_fun(const_ECEF, target_ECEF, an_alfa, an_beta)




