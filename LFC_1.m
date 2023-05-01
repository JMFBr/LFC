clear;clc;close all
%%

N_0  = 4; % #Planes
N_s0 = 1; % #Sats/planes
N_c  = 4; % Phasing parameter

L = [N_0, 0;
    N_c, N_s0]; % Lattice matrix

C = zeros(N_0, N_s0, 2); % Plane x Sat x Omega&M
%C(1,1,:) = 0; % RAAN and M of first satellite

figure()
hold on
for i = 1:N_0
    for j = 1:N_s0

        B = 2*pi*[i-1; j-1];
        C(i,j,:) = linsolve(L, B); % Matrix with all the pairs O-M

        Omega(i,j) = C(i,j,1);
        M(i,j)     = C(i,j,2);

        fun1 = @(m) (2*pi*(i-1)/N_0);        
        fun2 = @(m) ((2*pi*(j-1) - m*N_s0)/N_c);

        fplot(fun1,'LineWidth',1.5)
        fplot(fun2)        

    end
end
% xticks([ 0 pi 2*pi 3*pi])
% xticklabels({'0','\pi','2\pi','3\pi'})
% yticks([ 0 pi 2*pi 3*pi])
% yticklabels({'0','\pi','2\pi','3\pi'})
% xlim([0, 2*pi])
% ylim([0, 2*pi])
xlabel('M (rad)')
ylabel('\Omega (rad)')

C_deg = C*180/pi;

figure()
scatter(C(:,:,2), C(:,:,1))
grid on
ylabel('\Omega (rad)')
xlabel('M (rad)')

%% OEs

mu = 3.986e14; % [m3/s2], Earth standard gravitational parameter
RE = 6371e3;   % [m], Earth Radius
h  = 580e3;    % [m], Altitude

a  = RE + h;
e  = 0;
i  = 72*pi/180; % [rad], Inclination
om = 0*pi/180;  % [rad], argument of the perigee

OM = C(:,:,1); % [rad], random RAAN value from the LFC computation
M  = C(:,:,2); % [rad], random Mean Anomaly value from the LFC computation
th  = M; % Mean anomaly = to Eccentric anomaly = to true anomaly for e=0



th1 = 0;         % [rad]
th2 = 2*pi;      % [rad]
stepTh = pi/180; % [rad]

Terra_3D
for k = 1:size(th,1)
    for l = 1:size(th,2)

        [r,v] = kep2car(a, e, i, OM(k,l), om, th(k,l), mu);
        scatter3(r(1)/1000, r(2)/1000, r(3)/1000, 'filled', 'LineWidth',1.5)

        kepEl = [a, e, i, OM(k,l), om, th(k,l)];
        [X, Y, Z] = plotOrbit(kepEl, mu, th1, th2, stepTh);
        plot3(X./1000, Y./1000, Z./1000, 'Color','red')

    end

end









