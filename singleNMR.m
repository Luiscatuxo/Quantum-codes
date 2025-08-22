function [signal] = singleNMR(Bz,gh,gc,wRF,OmegaRF,intpoints,Jx1,Jx2,Jy1,Jy2,Jz1,Jz2,rhoin)

H0 = (gh*Bz)*Jz1/2 + (gc*Bz)*Jz2/2;   % We define the umperturbed hamiltonian
tpi2= 1/(4*OmegaRF);                  % This is the pi/2 pulse-time 

Id = [1  0 ; 0  1];
one = kron(Id,Id);     % We initialize the time evolution
Ut = one;

% This loop executes the time evolution and computes the evolution operator
for j=1:intpoints
tp = (j  )*tpi2/intpoints;
tm = (j-1)*tpi2/intpoints;
dt = tp-tm;
Hrot = OmegaRF*(Jx1 + Jx2)*cos((2*pi)*wRF*tp);
Up = expm(-1i*(2*pi)*(H0 + Hrot)*dt);
Ut = Up*Ut;
end

signal = real(trace(Ut*rhoin*Ut'* (Jz1 + Jz2) )); % We compute the expected value of Jz

end

