clear;

% This code simulates a pulse sequence experiment on a two-level quantum system.
% It models the qubit’s time evolution under a sequence of laser pulses
% plus a weak oscillating signal, then calculates the expectation value ⟨σₓ⟩
% as a function of the driving period.The final plot shows resonance when
% the driving matches the target signal frequency.

% We define the physical parameters
Odm = 4e7;              % Coupling of the driving during a pulse (Hz)
Ot = 2e4;               % Coupling of the signal we want to meassure (Hz)
tpi = 1/(2*Odm);        % Time of a laser pulse
wt = 2e6;               % Signal frecuency (Hz)

% We define the simulation parameters
points = 500;           % Number of frecuencies to sample
N = 200;                % Number of time points per pulse
delta = 1e6;            % Characteristic range
n = 100;                % Number of pulses

% We define the period grid
T_vec = linspace(1/(wt + delta), 1/(wt - delta), points);

% We define the Pauli matrices
sx = [0 1 ; 1 0];
sz = [1 0 ; 0 -1];

% We define the density matrix of the initial state |+>
p = (1/sqrt(2))*[1 ; 1];
m = (1/sqrt(2))*[1 ; -1];
g =[1 ; 0];
e =[0 ; 1];
rho = p*p';

for j = 1:points

    T = T_vec(j);     % We define the period of the driving
    U = eye(2);       % We initialize the propagator as the identity

    % We create the pulse sequence using the function
    [t_vec, pulses] = pulsesequence(T, n, Odm, N);
    
    for k = 1:n*N % This loop emulates the time evolution
        
        tp = t_vec(k);  % We select the instantaneous time
        dt = t_vec(2) - t_vec(1);
        
        % We compute the Hamiltonian in that time
        H = (pulses(k)/2) * sx + (Ot/2) * sz * cos(wt*tp);
        
        % We compute the time evolution operator
        U = expm(-1i *H * dt) * U;
      
    end
    
    % Now we compute <Sx> and save it 
    list_vals(j) = real(trace(U * rho * U' * sx));

end
figure;
plot(1./T_vec, list_vals, 'b', 'LineWidth', 1.2)
ylabel('\langle \sigma_{x} \rangle');
xlabel('1/T (Hz)');
xline(wt, '--black', 'LineWidth', 1.5, 'Label', '\omega_t');
grid off;