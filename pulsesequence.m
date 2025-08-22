function [t_vec, pulses] = pulsesquence(T, n, Amp, N)
    % Creates a signal with n pi pulses
    % T: Period of the signal
    % n: Number of pulses
    % Amp: Maximum amplitude of the pulse
    % N: Number of time points per pulse

    t_pi = 1 /(2*Amp);                  % Duration of the pi-pulses
    t_vec = linspace(0, n * T, n * N);  % Create a time array
    pulses = zeros(size(t_vec));        % Initialize the signal array
    T0 = (T - 2 * t_pi) ;               % Time shift to center pulses

    % Compute the pulse sequence
    for i = 1:length(t_vec)
        for j = 0:n-1
            if (t_vec(i) > (j * T + T0 / 2)) && (t_vec(i) < (j * T + T0 / 2 + t_pi))
                pulses(i) = Amp;
            elseif (t_vec(i) > ((j + 1) * T - T0 / 2 - t_pi)) && (t_vec(i) < ((j + 1) * T - T0 / 2))
                pulses(i) = Amp;
            end
        end
    end
end
