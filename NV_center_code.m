clear

% In this code we simulate an NV center stimulated with a static magnetic
% field in the z direction and a EM field in the x direction. We find a
% resonance when the frecuancy of the EM wave matches the transition
% frecuency of the NV center, producing a state transition and therefore a
% spin change.

% For the time simulation, since the hamiltonian is time dependent we make
% a trotterization technique meaning the hamiltonian is time independent in
% every time step, so we can apply the exponential of H as the time
% evolution operator.

% We define the physical parameters
Bz = 0.002;        % T
Bx = 0.000005;     % T
D = 2.87e9;        % Hz
ge = -28.05e9;     % Hz/T
w1 = D + ge * Bz;  % Resonance frecuency |1> Hz/T
w2 = D - ge * Bz;  % Resonance frecuency |-1> Hz/T

% We define the simulation parameters
points = 200;          % Number of frecuencies to sample
intpoints = 2000;      % Number of points for integrating one period (Because of the periodicity of the problem it is enough with integrating one period)
N = 100;               % Number of times we repeat the propagator associated to one period 
delta = (w1 - w2) / 2; % Characteristic range

w_vec = linspace(w2 - delta, w1 + delta, points); % We define the frecuency grid

% We define the Pauli matrices generalised for a 3 level system
sx = [0 1 0; 1 0 1; 0 1 0] / sqrt(2);
sz = [1 0 0; 0 0 0; 0 0 -1];

% We define the initial state as a density matrix
rho = [0 0 0; 0 1 0; 0 0 0];

for j = 1:points 

    w  = w_vec(j); % We select the driven frecuency in each  
    tf = 1 / w;    % We define the final time of one cicle of the driving 
    
    U = eye(3); % We initialize the propagator as the identity
    
    for k = 1:intpoints % This loop emulates the time evolutionh
        
        % We divide the interval from 0 to tf in a number 'intpoints' of steps
        tp = k * tf / intpoints;  
        tm = (k-1) * tf / intpoints;  
        dt = tp - tm;
        
        % We define the hamiltonian and compute the time evolution operator in that period
        H = D * sz^2 - ge * Bz * sz + sqrt(2) * ge * Bx * sx * cos( 2 * pi * w * tp);
        U = expm(-1i * (2 * pi) * H * dt) * U;
      
    end
    
    % We let evolve the system N periods 
    U = U^N; 
    
    % Now we compute <Sz> and save it 
    list_vals(j) = real(trace(U * rho * U' * sz));

end

% We plot the results
plot(w_vec, list_vals,'b')
ylabel('\langle S_z\rangle')
xlabel('\omega_d (Hz)')
xline(w1, '--r', 'LineWidth', 1.5);
xline(w2, '--k', 'LineWidth', 1.5); 
grid on