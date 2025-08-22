function [Jx1,Jx2,Jy1,Jy2,Jz1,Jz2,rhoin] = operators()

% We define the operators of the problem
Id = [1  0 ; 0  1];
Jx = [0  1 ; 1  0];
Jy = [0 -1i; 1i 0];
Jz = [1  0 ; 0 -1];

% We define the operators of the two qubit problem
Jx1 = kron(Jx,Id);
Jx2 = kron(Id,Jx);

Jy1 = kron(Jy,Id);
Jy2 = kron(Id,Jy);

Jz1 = kron(Jz,Id);
Jz2 = kron(Id,Jz);

e = [1;0]; % We define the excited state
g = [0;1]; % We define the ground state

e2 = kron(e,e); % We define the excited state of the 2 qubit system
g2 = kron(g,g); % We define the excited state of the 2 qubit system

% Also, the initial state
psin = e2;
rhoin = e2*e2'; % We initialize the system in the excited state

end

