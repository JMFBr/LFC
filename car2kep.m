function [a, e, i, OM, om, th] = car2kep(r, v, mu)

% car2kep.m - Conversion from Cartesian coordinates to Keplerian elements 
%
% PROTOTYPE:
% [a, e, i, OM, om, th] = car2kep(r, v, mu)
%
% DESCRIPTION:
% Conversion from Cartesian coordinates to Keplerian elements. Angles in
% radians.
%
% INPUTS:
% r          [3x1]   Position vector                               [km]
% v          [3x1]   Velocity vector                               [km/s]
%
% OPTIONAL INPUTS:
% mu         [1x1]   Gravitational parameter [km^3/s^2]
% When mu is not given, the algorithm uses the value 398600.433 km^3/s^2
%
% OUTPUTS:
% a          [1x1]   Semi-major axis                               [km]
% e          [1x1]   Eccentricity                                  [-]
% i          [1x1]   Inclination                                   [rad]
% OM         [1x1]   RAAN (right ascension of the ascending node)  [rad]
% om         [1x1]   Pericentre anomaly                            [rad]
% th         [1x1]   True anomaly                                  [rad]
%

if nargin < 2 % if not enough inputs, break
    error('At least 2 input arguments required.');
else
    if nargin < 3 % When mu is not given, the algorithm uses the value 398600.433 km^3/s^2
        mu = 398600.433;
    end
end

r_vec = r;      % Position vector 
v_vec = v;      % Velocity vector

r = norm (r);   % Position vector norm
v = norm (v);   % Velocity vector norm

h_vec = cross(r_vec,v_vec);      % Specific angular momentum (vector)
h = norm(h_vec);                 % Specific angular momentum (norm)

N_vec = cross([0;0;1],h_vec);    % Line of nodes (vector)
N = norm(N_vec);                 % Line of nodes (norm)

i = acos(h_vec(3)/norm(h));      % Inclination

e_vec = 1/mu* ( (v^2 - mu/r)* r_vec - (dot(r_vec,v_vec))* v_vec); % Eccentricity vector
e = norm(e_vec);   % Eccentricity 

a = 1/(2/r - v^2/mu); % Semi-major axis

% RAAN
if N_vec(2) >= 0
    OM = acos(N_vec(1) / N);
else
    OM = 2*pi - acos(N_vec(1) / N);
end

% Pericentre anomaly
if e_vec(3) >= 0
    om = acos(dot(N_vec,e_vec) / (N*e));
else
    om = 2*pi - acos(dot(N_vec,e_vec) / (N*e));
end

vr = dot(r_vec,v_vec)/r; % Radial velocity

% True anomaly
if vr >= 0
    th = acos(dot(e_vec,r_vec) / (e*r));
else
    th = 2*pi - acos(dot(e_vec,r_vec) / (e*r));
end
  
end

