function [X Y Z] = plotOrbit(kepEl, mu, th1, th2, stepTh)

% plotOrbit.m - Plot the arc length deltaTh of the orbit described by
% kepEl.
% PROTOTYPE:
% [X Y Z] = plotOrbit(kepEl, mu, deltaTh, stepTh)
%
% DESCRIPTION:
%   Plot the arc length of the orbit described by a set of orbital elements
%   for a specific arc length
%
% INPUT:
%   kepEl        [1x6]   orbital elements            [km, rad]
%   mu           [1x1]   gravitational parameter     [km^3/s^2]
%   deltaTh      [1x1]   arc length                  [rad]
%   stepTh       [1x1]   arc length st               [rad]
%
% OUTPUT:
%   X            [1xn]   X position                  [km]
%   Y            [1xn]   Y position                  [km]
%   Z            [1xn]   Z position                  [km]

%%%%%%%%%%%%%%%%

% kepEl = [a, e, i, OM, om, th]
% deltaTh = 2*pi for the entire orbit
% stepTh = pi/180 usually

th = th1:stepTh:th2;

a = kepEl(1);
e = kepEl(2);
i = kepEl(3);
OM = kepEl(4);
om = kepEl(5);
R = [];

for k = 1:length(th)
    th_ = th(k);
    [r,v] = kep2car(a, e, i, OM, om, th_, mu);
    R = [R r];
end

X = R(1,:);
Y = R(2,:);
Z = R(3,:);

end

