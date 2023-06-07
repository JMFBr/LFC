clear; clc; close all

%%

r_eci = [5.84755331e+06, 1.16128411e+06, -3.57406499e+06];
v_eci = [-4.09405652e+03, 1.96858544e+03, -6.05868299e+03];
utc = [2023 1 26 5 43 12];

[r_ecef, v_ecef] = eci2ecef(utc,r_eci,v_eci);

p = lla2ecef([84.89 60 0])
