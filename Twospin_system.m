% Program main
clear 

% This code simulates an NMR spectrum of a two–spin system (Hydrogen and Carbon-13)
% under a static magnetic field and a radiofrequency drive. It numerically computes
% the spin response as a function of the driving frequency and compares it with the
% corresponding analytical solution, showing resonance peaks at the nuclear Larmor frequencies.

% Program parameters
%--------------------------------------------------------
% (Some gyromagnetic ratios)
ge    =  28024.95266e6; % Hz*Tesla^-1
gh    =  42.577e6;      % Hydrogen     Hercios * Tesla^-1
gc    =  28.705e6;      % Carbono-13   Hercios * Tesla^-1

% (The next are control parameters, you can play with them)
Bz         = 0.01;     % Static magnetic field (in Teslas)
OmegaRF    = 2.5e4;    % Rabi frequency value (in Hz)
delta      = 60e4;     % This determines the values of frequency we will scan 
points     = 500;      % Points in the spectrum
intpoints  = 5000;     % This determines the accuracy of the numerical approach. 

%--------------------------------------------------------
[Jx1,Jx2,Jy1,Jy2,Jz1,Jz2,rhoin] = operators();
frec = linspace(0,8e5,points);
%--------------------------------------------------------

% Here we compute the NMR spectrum 
%--------------------------------------------------------
for j=1:points
 wRF = frec(j);
[signal] = singleNMR(Bz,gh,gc,wRF,OmegaRF,intpoints,Jx1,Jx2,Jy1,Jy2,Jz1,Jz2,rhoin);  %This is a subroutine (it is good to introduce subroutines
                                                                   %as they help to organise the program and make them more readable)
NMRsignal(j) = signal;
loop=j;

% Here the analytical solution (calculate it)
tpi2= 1/(4*OmegaRF);
delta1 = (gh*Bz - wRF);
delta2 = (gc*Bz - wRF);
gamma1 = sqrt((OmegaRF/2)^2 + (delta1/2)^2);
gamma2 = sqrt((OmegaRF/2)^2 + (delta2/2)^2);
beta1 = (OmegaRF/2)/gamma1;
beta2 = (OmegaRF/2)/gamma2;
alpha1 = (delta1/2)/gamma1;
alpha2 = (delta2/2)/gamma2;
s1 = (cos(2*pi*gamma1*tpi2))^2 + (alpha1^2 - beta1^2)*(sin(2*pi*gamma1*tpi2))^2;
s2 = (cos(2*pi*gamma2*tpi2))^2 + (alpha2^2 - beta2^2)*(sin(2*pi*gamma2*tpi2))^2;
Asolution(j)= s1 + s2;

end
hold on

plot(frec,NMRsignal,'LineWidth', 1.5); 
plot(frec,Asolution);
xline(gc*Bz,'--','label','\omega_C')
xline(gh*Bz,'--','label','\omega_H')
xlabel('Driving frecuency (Hz)');
ylabel('\langleJ_Z\rangle');
legend('Numerical solution', 'Analytic solution');

%--------------------------------------------------------
tpi2 = 1/(4*OmegaRF); % pi/2-pulse time 
accuracy = intpoints/(tpi2*gh*Bz);
%--------------------------------------------------------

