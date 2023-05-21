import numpy as np


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

(C, Omega, M, Omega_m, M_m) = LFC(N_0, N_s0, N_c)

# Create constellation matrix
const_matrix = np.ones((N_TS, 4))
const_matrix[:, 0] = a  # [m]
const_matrix[:, 1] = e
const_matrix[:, 2] = inc  # [rad]
const_matrix[:, 3] = om  # [rad]
const_matrix = np.c_[const_matrix, Omega, M]  # Constellation matrix: (Nts x 6 OEs)


## CONSTELLATION TO ECI:

def kep2eci(elements, mu_earth, t_0, t):  # R
    """
    Converts Keplerian orbital elements to Earth-centered inertial coordinates.
    Parameters:
    -----------
    elements : array_like
        Orbital elements in the following order: semi-major axis (a) [m], eccentricity (e),
        inclination (i) [rad], argument of periapsis (omega) [rad], right ascension of ascending node (Omega) [rad],
        mean anomaly (M) [rad].
    Returns:
    --------
    x_eci : array_like
        Earth-centered inertial coordinates in meters: x, y, z (m).
    """
    # Define Keplerian orbital elements
    a, e, i, omega, Omega = elements

    T = 2 * np.pi * np.sqrt(a ** 3 / mu_earth)  # Orbital period
    M = 2*np.pi*(t_0 + t)/T  # Mean anomaly at current position
    # Calculate eccentric anomaly
    E = M
    while True:
        E_new = E + (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        if abs(E_new - E) < 1e-8:
            break
        E = E_new
    if E < 0:
        E += 2*np.pi
    # Calculate true anomaly corresponding to the current time t
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    if nu < 0:
        nu += 2*np.pi
    # Calculate semi-latus rectum and mean motion
    p = a * (1 - e ** 2)
    # Calculate distance from Earth to satellite
    r = p / (1 + e * np.cos(nu))
    # Calculate position and velocity in perifocal coordinates
    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0])
    v_pqw = np.sqrt(mu_earth / p)*np.array([-np.sin(nu), e + np.cos(nu), 0])
    # Transformation matrix from perifocal to geocentric equatorial coordinates
    R_pqw_to_eci = np.array([
        [np.cos(Omega) * np.cos(omega) - np.sin(Omega) * np.sin(omega) * np.cos(i),
         -np.cos(Omega) * np.sin(omega) - np.sin(Omega) * np.cos(omega) * np.cos(i), np.sin(Omega) * np.sin(i)],
        [np.sin(Omega) * np.cos(omega) + np.cos(Omega) * np.sin(omega) * np.cos(i),
         -np.sin(Omega) * np.sin(omega) + np.cos(Omega) * np.cos(omega) * np.cos(i), -np.cos(Omega) * np.sin(i)],
        [np.sin(i) * np.sin(omega), np.sin(i) * np.cos(omega), np.cos(i)]
    ])
    # Convert
    x_eci = np.dot(R_pqw_to_eci, r_pqw)
    v_eci = np.dot(R_pqw_to_eci, v_pqw)

    return x_eci, v_eci


elements = const_matrix[0, 0:5]  # For satellite 1
t_0 = 0
t = 0

X_eci, V_eci = kep2eci(elements, mu, t_0, t)

## CONSTELLATION TO ECEF



## TARGET LIST INPUT AND TO ECEF

def read_targets():  # D
    lat_t, lon_t, weight = np.loadtxt("constellation_targets.csv", delimiter=',', usecols=(1, 2), unpack=True)

    # lon_t = lon_t + 180

    lat_t = np.radians(lat_t)
    lon_t = np.radians(lon_t)

    return lon_t, lat_t, weight

