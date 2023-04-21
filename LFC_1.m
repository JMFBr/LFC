clear;clc;close all
%%

N_0  = 4;  % #Planes
N_s0 = 3; % #Sats/planes
N_c  = 2; % Phasing parameter

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

