function [r,v] = kep2car(a, e, i, OM, om, th, mu)

% kep2car.m - Conversion from Keplerian elements to Cartesian coordinates 
%
% PROTOTYPE:
% [r, v] = kep2car(a, e, i, OM, om, th, mu)
%
% DESCRIPTION:
% Conversion from Keplerian elements to Cartesian coordinates. Angles in
% radians.
%
% INPUTS:
% a          [1x1]   Semi-major axis                               [km]
% e          [1x1]   Eccentricity                                  [-]
% i          [1x1]   Inclination                                   [rad]
% OM         [1x1]   RAAN (right ascension of the ascending node)  [rad]
% om         [1x1]   Pericentre anomaly                            [rad]
% th         [1x1]   True anomaly                                  [rad]
%
% OPTIONAL INPUTS:
% mu         [1x1]   Gravitational parameter                   [km^3/s^2]
% When mu is not given, the algorithm uses the value 398600.433 km^3/s^2
%
% OUTPUTS:
% r          [3x1]   Position vector                               [km]
% v          [3x1]   Velocity vector                               [km/s]

if nargin < 6
    error ('Check for missing input arguments')
else
    if nargin < 7   % When mu is not given, the algorithm uses the value 398600.433 km^3/s^2
        mu = 398600.433;
    end
end

p = a * (1 - e^2);                % Semi-latus rectum
r = p / (1 + e*cos(th) );         % Position vector norm

r_pf = r * [cos(th); sin(th); 0]; % Position vector in perifocal 
                                  % coordinate system
                                  
v_pf = sqrt(mu/p) * [-sin(th); e + cos(th); 0]; % Velocity vector in perifocal 
                                                % coordinate system
                                                

% Rotation matrix T:      Perifocal ----> ECI (Earth-centered inertial) %

R3_OM_t = [cos(OM), sin(OM), 0; -sin(OM), cos(OM), 0; 0, 0, 1]';
R1_i_t  = [1, 0, 0; 0, cos(i), sin(i); 0, -sin(i), cos(i)]';
R3_om_t = [cos(om), sin(om), 0; -sin(om), cos(om), 0; 0, 0, 1]';

T = R3_OM_t * R1_i_t * R3_om_t ;

r_ECI = T * r_pf;       r = r_ECI; % Position vector (ECI)

v_ECI = T * v_pf;       v = v_ECI; % Velocity vector (ECI)

end

