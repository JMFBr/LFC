clear;clc;close all
%%

N_TS = 44; 

N_0 = zeros(); 
j = 0;

for i = 1:N_TS
    if rem(N_TS,i) == 0
        j = j + 1;
        N_0(j) = i;
    end
end

N_s0 = N_TS./N_0; % #Sats/plane

for k = 1:length(N_0)              n                                         % LFC For all pairs N_0//N_s0 possible for 1 N_TS   
    
    N_c = [1:N_0(k)];

    for m = 1:length(N_c)                                                   % LFC for all N_c in the range

        L = [N_0(k), 0;
            N_c(m), N_s0(k)]; % Lattice matrix

        for i = 2:N_0(k)
            for j = 2:N_s0(k)

                B = 2*pi*[i-1; j-1];
                C(i,j,:) = linsolve(L, B);

                Omega(i,j) = C(i,j,1);
                M(i,j)     = C(i,j,2);


            end
        end

    end

end

