import numpy as np
import pandas as pd
from numpy import linalg as LA


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


## DATA
N_TS = 44  # Num satellites
N_0 = 4  # Num planes
N_s0 = 11  # Num satellites/planes
N_c = 4  # Phasing parameter

mu = 3.986e14  # [m3/s2], Earth standard gravitational parameter
RE = 6371e3  # [m], Earth Radius
h = 580e3  # [m], Altitude

a = RE + h
e = 0
inc = 72 * np.pi / 180  # [rad], Inclination
om = 0 * np.pi / 180  # [rad], argument of the perigee

# Sensor info (Simera)
h_s = 500e3  # [m], Altitude at which the sensor information is given
d_ac = 120e3  # [m], Swath width
d_al = 120e3

t_s = 24*3600


(C, Omega, M, Omega_m, M_m) = LFC(N_0, N_s0, N_c)


def kep2eci(const_m_OE):  # R modified for matrices
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
    M_v = const_m_OE[:, 5]  # Mean anomaly vector of satellite values

    # Calculate eccentric anomaly
    E = M_v
    nu = np.zeros(N_TS)
    for j in range(len(E)):  # Compute E using Newton's method for each satellite in the constellation
        while True:
            E_new = E[j] + (M_v[j] - E[j] + e_v[j] * np.sin(E[j])) / (1 - e_v[j] * np.cos(E[j]))
            if abs(E_new - E[j]) < 1e-8:
                break
            E[j] = E_new

        if E[j] < 0:
            E[j] += 2 * np.pi

        # Calculate true anomaly
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
    time_array: Date and time
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


def read_targets(time_array):
    """
    Import target list depending on the season
    """

    if time_array[1] <= 3:  # Jan to March included: winter
        target_m = pd.read_csv("winter.csv")
        print('Winter')
    if 4 <= time_array[1] <= 6:  # April to June: spring
        target_m = pd.read_csv("spring.csv")
        print('Spring')
    if 7 <= time_array[1] <= 9:  # July to Sep: summer
        target_m = pd.read_csv("fall.csv")
        print('Summer')
    if time_array[1] >= 10:  # Oct to Dec: fall
        target_m = pd.read_csv("fall.csv")
        print('Fall')

    target_m = target_m.to_numpy()  # Target matrix: Lat - Lon - Weight

    target_m[:, 0] = np.radians(target_m[:, 0])  # Latitude to radians
    target_m[:, 1] = np.radians(target_m[:, 1])  # Longitude to radians

    weights = target_m[:, 2]

    return target_m, weights


## LATLON TO ECEF
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


# CONSTELLATION
# Create constellation matrix with all satellites' orbital elements
const_OE = np.ones((N_TS, 4))
const_OE[:, 0] = a  # [m]
const_OE[:, 1] = e
const_OE[:, 2] = inc  # [rad]
const_OE[:, 3] = om  # [rad]
const_OE = np.c_[const_OE, Omega, M]  # Constellation matrix: (Nts x 6 OEs)

time_array_initial = np.array([2023, 6, 26, 5, 43, 12])
t = 0

T = 2 * np.pi * np.sqrt(a ** 3 / mu)  # [s], Orbital period
const_ECI = kep2eci(const_OE)
const_ECEF = eci2ecef(time_array_initial, const_ECI)


