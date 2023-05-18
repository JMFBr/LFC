import numpy as np
from numpy import linalg as LA


# Data
mu = 3.986e14  # [m3/s2], Earth standard gravitational parameter
RE = 6371e3  # [m], Earth Radius
h = 580e3  # [m], Altitude

a = RE + h  # [m], Semi-major axis
e = 0
inc = 72*np.pi/180  # [rad], Inclination
om = 0*np.pi/180  # [rad], Argument of the perigee


f_acr = 6.8*np.pi/180  # [rad],  From function solid angle
f_al = 3*np.pi/180  # [rad], Invented


def ra_and_dec_from_r(r):  # D
    # Calculates the right ascension (longitude) and the declination (latitude)
    # from the geocentric equatorial position vector

    l = r[0, :] / LA.norm(r, axis=0)  # direction cosine
    m = r[1, :] / LA.norm(r, axis=0)  # direction cosine
    n = r[2, :] / LA.norm(r, axis=0)  # direction cosine

    dec = np.arcsin(n)  # [rad] declination (latitude)

    ra = 2 * np.pi - np.arccos(l / np.cos(dec))  # [rad] right ascension (longitude)

    ra[m > 0] = np.arccos(l[m > 0] / np.cos(dec[m > 0]))  # [rad] right ascension (longitude)

    dec = np.degrees(dec)
    ra = np.degrees(ra)

    return ra, dec


# From Danilo's code, take out f_alo variable and along parameter
def g_t(a, r_rel, v_rel, f_acr):  # D
    # Rectangular coverage of the satellite: transform the points delimiting the coverage on earth to their lat and
    # lon coordinates

    v_ones = np.ones((1, 1, 1, 1))

    eta = a / RE

    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))  # Satellite across angle
    a_alfa = a_alfa.T * v_ones

    # Define unit vectors of the coverage RF:
    u_r = r_rel / LA.norm(r_rel, axis=0)
    u_v = v_rel / LA.norm(v_rel, axis=0)

    hh = np.cross(u_r, u_v, axisa=0, axisb=0, axisc=0)
    u_h = hh / LA.norm(hh, axis=0)
    yy = np.cross(u_h, u_r, axisa=0, axisb=0, axisc=0)
    u_y = yy / LA.norm(yy, axis=0)

    # Position vector of the delimiting points of the across coverage
    # -- M = Upper point
    # -- m = Lower point
    r_rel_M = np.cos(a_alfa) * u_r - np.sin(a_alfa) * u_h
    r_rel_M = r_rel_M / LA.norm(r_rel_M, axis=0)
    (ra_M, dec_M) = ra_and_dec_from_r(r_rel_M) # ra=RAAN=LONGITUDE, dec=declination=LATITUDE

    r_rel_m = np.cos(-a_alfa) * u_r - np.sin(-a_alfa) * u_h
    r_rel_m = r_rel_m / LA.norm(r_rel_m, axis=0)
    (ra_m, dec_m) = ra_and_dec_from_r(r_rel_m)

    print(ra_m.shape)

    return ra_M, ra_m, dec_M, dec_m


def unit_v(v):  # D
    u_v = v / LA.norm(v, axis=0)  # direction cosine

    return u_v


def dot_p(r_sat, r_t):  # D
    if np.ndim(r_sat) == 4:
        ang = np.einsum('mois,mt->tois', r_sat, r_t)
    elif np.ndim(r_sat) == 3:
        ang = np.einsum('mos,mt->tos', r_sat, r_t)
    elif np.ndim(r_sat) == 2:
        ang = np.einsum('ms,mt->ts', r_sat, r_t)

    # ang = np.einsum('mois,mt->tois', r_sat, r_t)

    return ang


def projections(r, v, r_t):  # D
    # Project Target coordinates into [ur, uh, uy] RF
    u_r = unit_v(r)
    u_v = unit_v(v)
    u_r_t = unit_v(r_t)

    print('new unit vectors calculated')

    u_h = np.cross(u_r, u_v, axisa=0, axisb=0, axisc=0)
    u_h = u_h / LA.norm(u_h, axis=0)
    u_y = np.cross(u_h, u_r, axisa=0, axisb=0, axisc=0)
    u_y = u_y / LA.norm(u_y, axis=0)

    print('new system reference calculated')

    # target projection on new system of reference

    p1 = dot_p(u_r, u_r_t)
    p2 = dot_p(u_y, u_r_t)
    p3 = dot_p(u_h, u_r_t)

    print('projections calculated')

    return p1, p2, p3


def filt_steps_fun(r, v, r_t, a_alfa):  # D
    dist_tol = 20  # [km] error tolerance in the cone sensor
    alf_tol = np.arctan(dist_tol / RE)

    p1, p2, p3 = projections(r, v, r_t)

    mask_p1 = p1 > 0  # Boolean, mask_p1(i)=True if p1(i)>0, p1=tr.ur must be >0 always

    # filt_steps_al = np.full(p1.shape, False)
    # filt_steps_ac = np.full(p1.shape, False)
    # filt_steps = np.full(p1.shape, False)

    # across track
    # phi = np.arctan2(p3, p1)
    # filt_steps_ac = np.absolute(phi) <= a_alfa

    filt_steps_ac = np.absolute(p3) / p1 <= np.tan(a_alfa - alf_tol)  # Boolean, True if tan(alpha_t)<=tan(alpha_s)
    filt_steps_ac[~mask_p1] = False  # Values in mask_p1 that correspond to False are set to False in filt_steps_ac

    print('across filter ok ', np.sum(filt_steps_ac))

    return filt_steps_ac


def filt_pop(a, r, v, r_t, f_acr):  # D
    eta = a / RE
    a_alfa = - f_acr + np.arcsin(eta * np.sin(f_acr))
    a_alfa = a_alfa.T

    filt_steps = filt_steps_fun(r, v, r_t, a_alfa)  # Boolean, True is target is covered
    cov_stepss = np.array(np.nonzero(filt_steps[:]))  # Return number of the Indices of filt_steps that are True aka
    # the covered targets

    return cov_stepss
