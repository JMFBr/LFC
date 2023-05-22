import numpy as np
import matplotlib.pyplot as plt

mu_Earth = 398600433000000.0  # Earth's gravitational parameter [m3 / s2]

# Initial conditions

a = 580.0e3 + 6371e3  # semi-major axis [m]
e = 0.0  # eccentricity
i = 72.0  # inclination [deg]
Omega = 342  # right ascension of the ascending node [deg]
omega = 0.0  # peri-apsis argument [deg]
nu_0 = 62.5  # Starting true anomaly [deg]

time_array_initial = (2023, 6, 26, 5, 43, 12)  # year, month, day, hour, minute, second (UTC)

# Simulation parameters
t_simulation = 0  # [s] time since the start of the simulation (initialised)
t_budget = 80  # [s] time budget
t_imaging = 7.5  # [s] imaging time
weights_flag = 0  # weights_flag = 0 corresponds to no weights considered in the simulation
plot_Sweden_flag = 1  # Plot_Sweden_Flag = 1 shows Sweden in the plot

# Satellite modelling
I = 0.0860  # [kg*m^2] maximum inertia
u = 0.0032  # [N*m] maximum control torque (reaction wheels)

# Targets (latitude, longitude, altitude, weight)
LocationsMatrix = np.array([[65.79440346,	23.28128957,	0,	2],
                            [58.57964914,	15.90509063,	0,	3],
                            [59.16054659,	14.18463314,	0,	6],
                            [59.20191132,	17.82478978,	0,	1],
                            [59.01827753,	12.28274628,	0,	5],
                            [59.01839185,	12.28269649,	0,	2],
                            [58.59414289,	16.11727729,	0,	3],
                            [62.97498879,	17.71738553,	0,	4],
                            [58.63685172,	16.23612391,	0,	1],
                            [60.06131545,	18.59513843,	0,	2],
                            [55.93064432,	14.32634457,	0,	1],
                            [65.31487106,	21.44049280,	0,	7],
                            [56.12075793,	13.14511603,	0,	2],
                            [56.74742532,	15.27257248,	0,	4],
                            [60.68028835,	17.26840842,	0,	5],
                            [58.70241593,	13.83472453,	0,	2],
                            [57.47025927,	15.51222412,	0,	3],
                            [63.32773977,	19.16212322,	0,	7],
                            [59.13970579,	12.91552131,	0,	3],
                            [62.39602138,	17.36344107,	0,	3],
                            [58.12234226,	12.12885250,	0,	1],
                            [65.27331407,	21.50622638,	0,	7],
                            [63.70264662,	20.32012089,	0,	1],
                            [60.68437573,	15.48385640,	0,	6],
                            [60.20435792,	16.30967066,	0,	4],
                            [56.99831601,	13.23547755,	0,	6],
                            [60.52209876,	15.41711399,	0,	3],
                            [56.04251357,	14.47618666,	0,	4],
                            [59.17981257,	12.55584717,	0,	7],
                            [57.97752698,	15.61525536,	0,	4],
                            [59.75820890,	12.35192913,	0,	7],
                            [63.27091482,	18.69737075,	0,	6],
                            [59.52933271,	15.44492104,	0,	5],
                            [59.79653538,	13.11760021,	0,	3],
                            [62.47796822,	17.33014533,	0,	1],
                            [58.75547557,	14.80794497,	0,	1],
                            [58.98928162,	12.26198673,	0,	5],
                            [60.64442303,	17.38668531,	0,	4],
                            [59.32175817,	13.44774481,	0,	4],
                            [57.09182308,	16.54752457,	0,	7],
                            [56.16051660,	14.76553876,	0,	2],
                            [57.22452832,	12.17747145,	0,	6],
                            [62.77226324,	17.92865811,	0,	4],
                            [57.48803616,	14.11573521,	0,	3],
                            [61.18594941,	17.17378596,	0,	3],
                            [57.84232206,	18.80447903,	0,	6],
                            [59.49957581,	16.02407193,	0,	3],
                            [65.56325384,	22.21546213,	0,	1],
                            [58.67757264,	17.13264696,	0,	6],
                            [61.04989683,	15.19673449,	0,	2],
                            [60.90444766,	15.16734750,	0,	6],
                            [61.25962446,	17.16939660,	0,	4],
                            [59.35305313,	18.10271083,	0,	2],
                            [55.93069485,	14.32551430,	0,	1],
                            [58.19731491,	13.64018584,	0,	2],
                            [58.77265837,	16.82075514,	0,	2],
                            [56.89075885,	12.48010732,	0,	1],
                            [59.73081581,	14.23395719,	0,	6],
                            [60.98163425,	15.25814694,	0,	1],
                            [60.93073326,	15.09776047,	0,	3],
                            [58.67757264,	17.13264696,	0,	3],
                            [60.90444766,	15.16734750,	0,	5],
                            [57.87864079,	18.97914858,	0,	3],


[	59.3294	,	18.0686	,	0	,	1	]	,
[	57.7075	,	11.9675	,	0	,	2	]	,
[	55.6058	,	13.0358	,	0	,	6	]	,
[	59.8581	,	17.6447	,	0	,	6	]	,
[	58.6	,	16.2	,	0	,	5	]	,
[	59.8601	,	17.64	,	0	,	3	]	,
[	59.6161	,	16.5528	,	0	,	3	]	,
[	59.2739	,	15.2075	,	0	,	4	]	,
[	58.4158	,	15.6253	,	0	,	1	]	,
[	56.05	,	12.7167	,	0	,	7	]	,
[	57.7828	,	14.1606	,	0	,	1	]	,
[	57.6667	,	15.85	,	0	,	6	]	,
[	62.4	,	17.3167	,	0	,	2	]	,
[	60.6747	,	17.1417	,	0	,	5	]	,
[	63.825	,	20.2639	,	0	,	4	]	,
[	64.75	,	20.95	,	0	,	3	]	,
[	59.3783	,	13.5042	,	0	,	2	]	,
[	59.1958	,	17.6281	,	0	,	3	]	,
[	56.6739	,	12.8572	,	0	,	3	]	,
[	59.3708	,	16.5097	,	0	,	1	]	,
[	56.1608	,	15.5861	,	0	,	3	]	,
[	56.8769	,	14.8092	,	0	,	3	]	,
[	57.7211	,	12.9403	,	0	,	5	]	,
[	59.4333	,	18.0833	,	0	,	6	]	,
[	58.2828	,	12.2892	,	0	,	4	]	,
[	63.1792	,	14.6358	,	0	,	1	]	,
[	65.5844	,	22.1539	,	0	,	2	]	,
[	59.5167	,	17.9167	,	0	,	1	]	,
[	60.4856	,	15.4364	,	0	,	1	]	,
[	55.3667	,	13.1667	,	0	,	3	]	,
[	56.6614	,	16.3628	,	0	,	4	]	,
[	58.5	,	13.1833	,	0	,	4	]	,
[	58.3833	,	13.85	,	0	,	2	]	,
[	58.7531	,	17.0086	,	0	,	7	]	,
[	57.93	,	12.5331	,	0	,	3	]	,
[	59.2	,	17.8167	,	0	,	1	]	,
[	60.6072	,	15.6311	,	0	,	1	]	,
[	57.65	,	12.0167	,	0	,	1	]	,
[	57.75	,	16.6333	,	0	,	2	]	,
[	57.1167	,	12.2167	,	0	,	1	]	,
[	58.35	,	11.9167	,	0	,	4	]	,
[	55.8706	,	12.8311	,	0	,	6	]	,
[	63.2908	,	18.7156	,	0	,	7	]	,
[	55.7939	,	13.1133	,	0	,	3	]	,
[	56.0337	,	14.1333	,	0	,	1	]	,
[	59.3667	,	18.15	,	0	,	3	]	,
[	58.5333	,	15.0333	,	0	,	7	]	,
[	59.5333	,	18.0833	,	0	,	2	]	,
[	55.4167	,	13.8333	,	0	,	1	]	,
[	59.4833	,	18.3	,	0	,	6	]	,
[	56.9053	,	12.4911	,	0	,	3	]	,
[	59.3333	,	14.5167	,	0	,	2	]	,
[	59.6167	,	17.85	,	0	,	7	]	,
[	67.8489	,	20.3028	,	0	,	2	]	,
[	57.629	,	18.3071	,	0	,	3	]	,
[	59	,	16.2	,	0	,	3	]	,
[	59.3167	,	18.25	,	0	,	4	]	,
[	58.3806	,	12.325	,	0	,	2	]	,
[	65.3333	,	21.5	,	0	,	3	]	,
[	57.4833	,	12.0667	,	0	,	3	]	,
[	60.6167	,	16.7833	,	0	,	4	]	,
[	57.8667	,	11.9667	,	0	,	4	]	,
[	57.7919	,	14.2756	,	0	,	7	]	,
[	59.1167	,	15.1333	,	0	,	1	]	,
[	59.6356	,	17.0764	,	0	,	1	]	,
[	55.6333	,	13.7167	,	0	,	4	]	,
[	56.1667	,	14.85	,	0	,	3	]	,
[	58.3833	,	13.4333	,	0	,	1	]	,
[	62.6361	,	17.9411	,	0	,	6	]	,
[	56.1667	,	13.7667	,	0	,	2	]	,
[	59.3	,	14.1167	,	0	,	1	]	,
[	55.8392	,	13.3039	,	0	,	7	]	,
[	59.5167	,	15.9833	,	0	,	4	]	,
[	59.7667	,	18.7	,	0	,	2	]	,
[	57.265	,	16.45	,	0	,	7	]	,
[	57.7667	,	12.3	,	0	,	6	]	,
[	65.8256	,	21.6906	,	0	,	1	]	,
[	56.1167	,	13.15	,	0	,	4	]	,
[	57.65	,	14.6833	,	0	,	1	]	,
[	58.7	,	13.8167	,	0	,	4	]	,
[	58.175	,	13.5531	,	0	,	3	]	,
[	56.2	,	12.5667	,	0	,	7	]	,
[	59.2	,	17.9	,	0	,	3	]	,
[	57.6667	,	12.1167	,	0	,	1	]	,
[	57.5167	,	12.6833	,	0	,	5	]	,
[	59.3667	,	17.0333	,	0	,	6	]	,
[	55.55	,	12.9167	,	0	,	2	]	,
[	56.8333	,	13.9333	,	0	,	6	]	,
[	59.6167	,	16.25	,	0	,	2	]	,
[	59.1167	,	18.0667	,	0	,	6	]	,
[	61.7333	,	17.1167	,	0	,	3	]	,
[	55.6333	,	13.2	,	0	,	1	]	,
[	60.1333	,	15.1833	,	0	,	3	]	,
[	59.6542	,	12.5914	,	0	,	3	]	,
[	58.0333	,	14.9667	,	0	,	5	]	,
[	58.3333	,	15.1167	,	0	,	2	]	,
[	55.6667	,	13.0833	,	0	,	3	]	,
[	56.7333	,	15.9	,	0	,	2	]	,
[	58.9	,	17.95	,	0	,	2	]	,
[	59.3939	,	15.8386	,	0	,	6	]	,
[	61.3481	,	16.3947	,	0	,	7	]	,
[	59.3333	,	13.4333	,	0	,	2	]	,
[	59.5833	,	17.5	,	0	,	6	]	,
[	57.4333	,	15.0667	,	0	,	5	]	,
[	59.8333	,	13.1333	,	0	,	7	]	,
[	61.0167	,	14.5333	,	0	,	2	]	,
[	58.7	,	15.8	,	0	,	5	]	,
[	59.9167	,	16.6	,	0	,	1	]	,
[	60.0333	,	13.65	,	0	,	4	]	,
[	56.2	,	15.2833	,	0	,	4	]	,
[	60.1456	,	16.1683	,	0	,	3	]	,
[	61.3	,	17.0833	,	0	,	5	]	,
[	59.2833	,	17.8	,	0	,	2	]	,
[	58.6667	,	17.1167	,	0	,	7	]	,
[	57.7833	,	13.4167	,	0	,	1	]	,
[	59.3333	,	18.3833	,	0	,	1	]	,
[	60.35	,	15.75	,	0	,	6	]	,
[	60.0042	,	15.7933	,	0	,	1	]	,
[	57.5667	,	12.1	,	0	,	4	]	,
[	59.7167	,	14.1667	,	0	,	2	]	,
[	55.5	,	13.2333	,	0	,	6	]	,
[	55.4167	,	12.95	,	0	,	5	]	,
[	62.4869	,	17.3258	,	0	,	1	]	,
[	60.6	,	15.0833	,	0	,	1	]	,
[	56.2833	,	13.2833	,	0	,	3	]	,
[	59.15	,	18.1333	,	0	,	4	]	,
[	57.5833	,	11.9333	,	0	,	7	]	,
[	55.6333	,	13.0833	,	0	,	2	]	,
[	57.3	,	13.5333	,	0	,	3	]	,
[	59.25	,	18.1833	,	0	,	6	]	,
[	58.0833	,	11.8167	,	0	,	4	]	,
[	57.9	,	12.0667	,	0	,	4	]	,
[	57.6669	,	14.9703	,	0	,	6	]	,
[	59.8167	,	17.7	,	0	,	1	]	,
[	59.5833	,	15.25	,	0	,	5	]	,
[	55.7167	,	13.0167	,	0	,	1	]	,
[	59.2861	,	18.2872	,	0	,	6	]	,
[	56.1347	,	12.9472	,	0	,	6	]	,
[	55.9167	,	14.2833	,	0	,	4	]	,
[	59.4833	,	17.75	,	0	,	3	]	,
[	59.05	,	12.7	,	0	,	3	]	,
[	59.1333	,	12.9333	,	0	,	5	]	,
[	56.55	,	14.1333	,	0	,	3	]	,
[	59.35	,	18.2	,	0	,	1	]	,
[	57.9167	,	14.0667	,	0	,	1	]	,
[	63.1667	,	17.2667	,	0	,	3	]	,
[	58.4167	,	14.1667	,	0	,	7	]	,
[	64.6	,	18.6667	,	0	,	4	]	,
[	67.13	,	20.66	,	0	,	3	]	,
[	59.6167	,	17.7167	,	0	,	1	]	,
[	56.0442	,	14.5753	,	0	,	5	]	,
[	58.1833	,	13.95	,	0	,	4	]	,
[	56.1344	,	13.1283	,	0	,	3	]	,
[	57.85	,	14.1167	,	0	,	3	]	,
[	57.8167	,	12.3667	,	0	,	6	]	,
[	56.9	,	14.55	,	0	,	4	]	,
[	58.2833	,	11.4333	,	0	,	1	]	,
[	55.9361	,	13.5472	,	0	,	2	]	,
[	59.5	,	13.3167	,	0	,	5	]	,
[	59.4167	,	16.4667	,	0	,	7	]	,
[	56.05	,	14.4667	,	0	,	6	]	,
[	55.4833	,	13.5	,	0	,	3	]	,
[	59.175	,	17.4333	,	0	,	5	]	,
[	59.2917	,	18.2528	,	0	,	3	]	,
[	58.9333	,	11.1667	,	0	,	1	]	,
[	57.6	,	12.05	,	0	,	5	]	,
[	56.2667	,	14.5333	,	0	,	4	]	,
[	65.85	,	23.1667	,	0	,	5	]	,
[	60.2769	,	15.9872	,	0	,	3	]	,
[	59.1805	,	18.1804	,	0	,	7	]	,
[	59.2361	,	14.4297	,	0	,	5	]	,
[	56.3667	,	13.9833	,	0	,	3	]	,
[	57.6833	,	12.2	,	0	,	6	]	,
[	56.1333	,	13.3833	,	0	,	6	]	,
[	59.0667	,	15.1167	,	0	,	4	]	,
[	55.85	,	13.65	,	0	,	5	]	,
[	59.7167	,	17.8	,	0	,	1	]	,
[	60.65	,	17.0333	,	0	,	7	]	,
[	59.5167	,	17.65	,	0	,	6	]	,
[	58.4833	,	16.3167	,	0	,	3	]	,
[	60.0833	,	15.95	,	0	,	3	]	,
[	55.4	,	12.85	,	0	,	2	]	,
[	58.2	,	16	,	0	,	2	]	,
[	56.0833	,	12.9167	,	0	,	1	]	,
[	55.4667	,	13.0167	,	0	,	2	]	,
[	60.55	,	16.2833	,	0	,	1	]	,
[	58.5083	,	15.5028	,	0	,	2	]	,
[	59.1903	,	18.1275	,	0	,	4	]	,
[	65.8333	,	24.1333	,	0	,	2	]	,
[	55.55	,	14.35	,	0	,	6	]	,
[	56.5167	,	13.0333	,	0	,	4	]	,
[	59.5167	,	15.0333	,	0	,	4	]	,
[	63.15	,	14.75	,	0	,	3	]	,
[	55.9667	,	12.7667	,	0	,	7	]	,
[	55.55	,	13.95	,	0	,	6	]	,
[	58.3	,	14.2833	,	0	,	7	]	,
[	59.9667	,	17.7	,	0	,	7	]	,
[	57.1667	,	15.3333	,	0	,	5	]	,
[	55.7667	,	13.0167	,	0	,	7	]	,
[	61.8333	,	16.0833	,	0	,	5	]	,


                            ])


# Initial calculations

T = 2*np.pi*np.sqrt(a ** 3 / mu_Earth)  # Orbital period
E_0 = 2*np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(np.deg2rad(nu_0) / 2))  # Initial eccentric anomaly
if E_0 < 0:
    E_0 += 2*np.pi
M_nu0 = E_0 - e*np.sin(E_0)  # Initial mean anomaly
t_nu0 = M_nu0 * T / (2*np.pi)  # Time from peri-apsis that corresponds to the initial true anomaly

elements_array = a, e, i, omega, Omega


def kep2eci(elements, mu_earth, t_0, t):
    """
    Converts Keplerian orbital elements to Earth-centered inertial coordinates.

    Parameters:
    -----------
    elements : array_like
        Orbital elements in the following order: semi-major axis (a) [m], eccentricity (e),
        inclination (i) [deg], argument of periapsis (omega) [deg], right ascension of ascending node (Omega) [deg],
        mean anomaly (M) [deg].

    Returns:
    --------
    x_eci : array_like
        Earth-centered inertial coordinates in meters: x, y, z (m).
    """

    # Define Keplerian orbital elements
    a, e, i, omega, Omega = elements

    # Convert orbital elements to radians
    i = np.deg2rad(i)
    omega = np.deg2rad(omega)
    Omega = np.deg2rad(Omega)

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
    # v_pqw = np.sqrt(mu_earth / p)*np.array([-np.sin(nu), e + np.cos(nu), 0])

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

    return x_eci


def geo2eci(lat, lon, alt, time_array, t):
    """
    Converts geodetic coordinates to Earth-centered, Earth-fixed coordinates.

    Parameters:
    -----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt : float
        Altitude in meters.
    time_array : array
        date and time at the start of the imaging window

    Returns:
    --------
    x_eci : array_like
        Earth-centered inertial coordinates in meters: x, y, z (m).
    """

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    # Define WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis (m)
    b = 6356752.0  # semi-minor axis (m)

    f = 1 - b / a  # flattening of Earth's ellipsoid
    e2 = 1 - b ** 2 / a ** 2  # square of the first numerical eccentricity of Earth's ellipsoid

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    x_ecef = (N + alt) * np.cos(lat) * np.cos(lon)
    y_ecef = (N + alt) * np.cos(lat) * np.sin(lon)
    z_ecef = ((1 - f) ** 2 * N + alt) * np.sin(lat)

    Y = time_array[0]  # year
    M = time_array[1]  # month
    D = time_array[2]  # day
    ho = time_array[3]  # hour (in UTC time)
    mi = time_array[4] + np.floor(t / 60)  # minutes (in UTC time), adding the time since the start of the imaging
    se = time_array[5] + t % 60  # seconds (in UTC time)

    jd = 1721013.5 + 367 * Y - int(7 / 4 * (Y + int((M + 9) / 12))) + int(275 * M / 9) + D + (
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

    # Rotation
    x_eci_array = np.dot(R_ECEF2ECI, np.array([x_ecef, y_ecef, z_ecef]))

    return x_eci_array


def angleBetweenTargets(sat_pos_ECI_current, target1, target2, time_array_initial, t_current):

    loc_ECI1 = geo2eci(target1[0], target1[1], target1[2],
                       time_array_initial, t_current)
    # print(loc_ECI1/1000)
    r_1S = sat_pos_ECI_current - loc_ECI1

    loc_ECI2 = geo2eci(target2[0], target2[1], target2[2],
                       time_array_initial, t_current)
    r_2S = sat_pos_ECI_current - loc_ECI2

    return np.arccos(np.dot(r_1S / np.linalg.norm(r_1S), r_2S / np.linalg.norm(r_2S)))


def timeFromAngle(phi):

    # t = float(np.sqrt(4*I*phi/u))

    x = np.rad2deg(float(phi))
    t = -1.79096049749581e-12 * x ** 12 + 3.22473469945933e-10 * x ** 11 - 2.55154917035896e-08 * x ** 10 +	\
        1.16661848602749e-06 * x ** 9 - 3.40981170519613e-05 * x ** 8 + 0.000665709591093221 * x ** 7 - \
        0.00881318029653716 * x ** 6 + 0.0787423082028696 * x ** 5 - 0.464119676655089 * x ** 4 + \
        1.72771212613387 * x ** 3 - 3.78080965401434 * x ** 2 + 4.54896778268871 * x + 0.122704889685351

    return t


# Calculate principal angle from target to satellite's nadir (initial target choice)
array_target_nadir_angle = np.zeros((len(LocationsMatrix[:, 0]), 1))
sat_pos_ECI = kep2eci(elements_array, mu_Earth, t_nu0, t_simulation)

for k in range(len(LocationsMatrix[:, 0])):
    loc_ECI = geo2eci(LocationsMatrix[k, 0], LocationsMatrix[k, 1], LocationsMatrix[k, 2], time_array_initial,
                      t_simulation)
    r_I_ST = (sat_pos_ECI - loc_ECI)/np.linalg.norm(sat_pos_ECI - loc_ECI)
    array_target_nadir_angle[k] = np.arccos(np.dot(loc_ECI/np.linalg.norm(loc_ECI), r_I_ST))


# Calculate the highest collected value for the selection of the first target
array_value_first = np.zeros((len(LocationsMatrix[:, 0]), 1))

for k in range(len(LocationsMatrix[:, 0])):

    if weights_flag == 0:
        array_value_first[k] = np.cos(array_target_nadir_angle[k]) ** 2

    else:
        array_value_first[k] = np.cos(array_target_nadir_angle[k]) ** 2 * LocationsMatrix[k, 3]

first_target_index = np.argmax(array_value_first)

# Simulation time propagation for the first target
t_simulation = timeFromAngle(array_target_nadir_angle[first_target_index]) + t_imaging

# Collected value
cumulative_value = np.max(array_value_first)

previous_target = LocationsMatrix[np.argmax(array_value_first), :]

# Removing the scheduled targets from the locations list
Unscheduled_LocationsMatrix = np.concatenate((LocationsMatrix[0:first_target_index, :],
                                             LocationsMatrix[first_target_index + 1:, :]))

Scheduled_LocationsMatrix = previous_target

# Scheduling
while t_simulation < t_budget:

    # Initializing value array for all targets
    array_value = np.zeros((len(Unscheduled_LocationsMatrix[:, 0]), 1))

    # Calculating current satellite position
    sat_pos_ECI_current = kep2eci(elements_array, mu_Earth, t_nu0, t_simulation)
    # print(sat_pos_ECI_current)

    # Calculating values for all targets
    for j in range(len(Unscheduled_LocationsMatrix[:, 0])):

        loc_ECI = geo2eci(Unscheduled_LocationsMatrix[j, 0], Unscheduled_LocationsMatrix[j, 1],
                          Unscheduled_LocationsMatrix[j, 2], time_array_initial, t_simulation)

        r_I_ST = (sat_pos_ECI_current - loc_ECI) / np.linalg.norm(sat_pos_ECI_current - loc_ECI)
        off_nadir_angle = np.arccos(np.dot(sat_pos_ECI_current / np.linalg.norm(sat_pos_ECI_current), r_I_ST))

        if weights_flag == 0:
            array_value[j] = np.cos(off_nadir_angle) ** 2

        else:
            array_value[j] = np.cos(off_nadir_angle) ** 2 * Unscheduled_LocationsMatrix[j, 3]

    next_target_index = np.argmax(array_value)
    # print(np.rad2deg(np.arccos(np.sqrt(array_value[next_target_index]))))
    next_target = Unscheduled_LocationsMatrix[next_target_index, :]
    # print(next_target)

    principal_angle = angleBetweenTargets(sat_pos_ECI_current, previous_target, next_target,
                                          time_array_initial, t_simulation)

    # print(np.rad2deg(principal_angle))

    # Time propagation
    t_simulation = t_simulation + timeFromAngle(principal_angle) + t_imaging

    if t_simulation < t_budget:

        # Removing the scheduled targets from the locations list
        Unscheduled_LocationsMatrix = np.concatenate((Unscheduled_LocationsMatrix[0:next_target_index, :],
                                                      Unscheduled_LocationsMatrix[next_target_index + 1:, :]))

        # Updating the list of scheduled targets
        Scheduled_LocationsMatrix = np.vstack((Scheduled_LocationsMatrix, next_target))

        # Add value
        cumulative_value = cumulative_value + array_value[next_target_index]

        previous_target = next_target


Scheduled_LocationsMatrix_NoWeights = Scheduled_LocationsMatrix[:, 0:3]
# print(Scheduled_LocationsMatrix_NoWeights)

print('Scheduled targets:', len(Scheduled_LocationsMatrix[:, 0]))
print('Total imaging time:', round(float(t_simulation - timeFromAngle(principal_angle) - t_imaging), 3), 's')
print('Total value:', round(float(cumulative_value), 3))


# Plots

Plot_ECI_scheduled_locations = np.zeros((len(Scheduled_LocationsMatrix[:, 0]), 3))

for w in range(len(Plot_ECI_scheduled_locations[:, 0])):
    Plot_ECI_scheduled_locations[w, :] = geo2eci(Scheduled_LocationsMatrix[w, 0], Scheduled_LocationsMatrix[w, 1],
                                                 Scheduled_LocationsMatrix[w, 2], time_array_initial, 0)


Plot_ECI_unscheduled_locations = np.zeros((len(Unscheduled_LocationsMatrix[:, 0]), 3))

for z in range(len(Plot_ECI_unscheduled_locations[:, 0])):
    Plot_ECI_unscheduled_locations[z, :] = geo2eci(Unscheduled_LocationsMatrix[z, 0], Unscheduled_LocationsMatrix[z, 1],
                                                   Unscheduled_LocationsMatrix[z, 2], time_array_initial, 0)


fig = plt.figure()
ax = plt.axes(projection='3d')


def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc...  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


xdata_unscheduled = Plot_ECI_unscheduled_locations[:, 0]
ydata_unscheduled = Plot_ECI_unscheduled_locations[:, 1]
zdata_unscheduled = Plot_ECI_unscheduled_locations[:, 2]
ax.scatter3D(xdata_unscheduled, ydata_unscheduled, zdata_unscheduled, color='c', label=r'Unscheduled targets')

xdata_scheduled = Plot_ECI_scheduled_locations[:, 0]
ydata_scheduled = Plot_ECI_scheduled_locations[:, 1]
zdata_scheduled = Plot_ECI_scheduled_locations[:, 2]
ax.scatter3D(xdata_scheduled, ydata_scheduled, zdata_scheduled, color='g', label=r'Scheduled targets')

xline = np.array(Plot_ECI_scheduled_locations[:, 0])
yline = np.array(Plot_ECI_scheduled_locations[:, 1])
zline = np.array(Plot_ECI_scheduled_locations[:, 2])
ax.plot3D(xline, yline, zline, color='g')

set_axes_equal(ax)

ax.set_xlabel(r'$X\ \ [m]$')
ax.set_ylabel(r'$Y\ \ [m]$')
ax.set_zlabel(r'$Z\ \ [m]$')

plt.legend(loc='upper right')


if plot_Sweden_flag == 1:

    SwedenBorder = np.array([   [20.104980, 69.029279, 0],
                                [19.929199, 68.358699, 0],
                                [18.391113, 68.552351, 0],
                                [17.863770, 67.958148, 0],
                                [17.248535, 68.114293, 0],
                                [16.259766, 67.516972, 0],
                                [15.490723, 66.293373, 0],
                                [14.589844, 66.142743, 0],
                                [13.732910, 64.595613, 0],
                                [14.084473, 64.024122, 0],
                                [12.744141, 64.024122, 0],
                                [11.953125, 63.322549, 0],
                                [12.150879, 61.700291, 0],
                                [12.744141, 61.079544, 0],
                                [12.304688, 60.994423, 0],
                                [12.346387, 60.272515, 0],
                                [11.477344, 59.523176, 0],
                                [11.513672, 58.881942, 0],
                                [11.093945, 59.051856, 0],
                                [11.617480, 57.844751, 0],
                                [12.519922, 56.607885, 0],
                                [12.546387, 56.231139, 0],
                                [12.885840, 55.316544, 0],
                                [14.304199, 55.416192, 0],
                                [14.523926, 56.108810, 0],
                                [15.864258, 56.121060, 0],
                                [16.911328, 57.385783, 0],
                                [16.743164, 57.704147, 0],
                                [17.102891, 58.556792, 0],
                                [18.457031, 59.125226, 0],
                                [18.973389, 59.971508, 0],
                                [17.369385, 60.716198, 0],
                                [17.138672, 61.653379, 0],
                                [18.555908, 63.059937, 0],
                                [20.665283, 63.806743, 0],
                                [21.544189, 64.482261, 0],
                                [21.126709, 64.816233, 0],
                                [22.554932, 65.811781, 0],
                                [24.147949, 65.771236, 0],
                                [23.642578, 66.443107, 0],
                                [24.027100, 66.852446, 0],
                                [23.785400, 67.407487, 0],
                                [23.565674, 67.999341, 0],
                                [20.104980, 69.029279, 0],
                                ])

    GotlandBorder = np.array([  [18.182373, 56.909002, 0],
                                [19.017334, 57.379861, 0],
                                [18.919580, 57.710017, 0],
                                [19.535937, 58.184808, 0],
                                [18.479004, 57.827205, 0],
                                [18.083496, 57.521723, 0],
                                [18.182373, 56.909002, 0],
                                ])

    Plot_Sweden = np.zeros((len(SwedenBorder[:, 0]), 3))

    for w in range(len(SwedenBorder[:, 0])):
        Plot_Sweden[w, :] = geo2eci(SwedenBorder[w, 1], SwedenBorder[w, 0],
                                    SwedenBorder[w, 2], time_array_initial, 0)

    Plot_Gotland = np.zeros((len(GotlandBorder[:, 0]), 3))

    for w in range(len(GotlandBorder[:, 0])):
        Plot_Gotland[w, :] = geo2eci(GotlandBorder[w, 1], GotlandBorder[w, 0],
                                     GotlandBorder[w, 2], time_array_initial, 0)

    xSweden = np.array(Plot_Sweden[:, 0])
    ySweden = np.array(Plot_Sweden[:, 1])
    zSweden = np.array(Plot_Sweden[:, 2])
    ax.plot3D(xSweden, ySweden, zSweden, color='k')

    xGotland = np.array(Plot_Gotland[:, 0])
    yGotland = np.array(Plot_Gotland[:, 1])
    zGotland = np.array(Plot_Gotland[:, 2])
    ax.plot3D(xGotland, yGotland, zGotland, color='k')


plt.show()



