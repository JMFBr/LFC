clear; clc; close all

%%

DV_m_r = [11., 11., 11., 22.;
       4.,  4.,  4.,  2.;
       1.,  2.,  3.,  1.];

num_targets_mean = [216.47258904, 372.81252501, 495.92537015, 582.09663866];

figure()
scatter3(DV_m_r(1, :), DV_m_r(2, :), DV_m_r(3, :), 30, num_targets_mean, 'filled')
colorbar
grid on
xlim([0, 25])
ylim([0, 5])
zlim([0, 5])
xlabel('N_s0');
ylabel('N_0');
zlabel('N_c');
title('Avg targets per time step');