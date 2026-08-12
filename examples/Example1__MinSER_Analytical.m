% JSTQE - PAM6 Design for IM-DD with RIN
gpuDevice(1);
addpath("../functions");
addpath("../src");

% === Simulation parameters ===
% > PAM-6 <
M = 6;
Nb = 2.5;
X_set = -(M-1):2:(M-1);
X_set_2D = X_set + 1j*X_set';
Px_2D = 1/32*ones(6,6);
Px = 1/M*ones(1,M);
Px_cmf = filter(1, [1 -1], Px);     % sum P(x)

% > IM-DD parameters <
er = db2pow(4.5);               % Extinction Ratio in dB
bias = (M-1)*(er+1)/(er-1);     % IM Bias
N0_th = (22e-12)^2;             % One-sided PSD of thermal noise in [pA/sqrt(Hz)]
N0_rin = db2pow(-144);          % One-sided PSD of RIN in [dB/Hz]
L = 0.5;                        % Fiber length in [km]

% > Bitrate, Coderate, Symbol rate <
Rb = 600e9;
Rc = 514/544*120/128;           % Concatenated KP4+Hamming
%Rc = 514/544;                   % Standalone KP4
Rs = Rb/(Rc*2.5);
BW = Rs/2;

% > MC simulation params
oma_dBm_vec = -10:1:10;
oma_dBm = 0;
N_oma = length(oma_dBm_vec);
Nsym = 1e5;

% Indices for symbol removal in QAM-36 -> QAM-32
% load('QAM36_idx2rmv.mat');          % Reduced set (ECOC 2025)
load('QAM36_idx2rmv_ext.mat');      % Full set (JSTQE 2026)
N_ix = length(Cmb);

%% (1.1) SNR Maximization: Analytical

snr_all = zeros(N_ix, N_oma);

% Loop over constellations
for ii = 1:N_ix
    ix_rmv = Cmb(ii,:);
    fprintf("--- Combination %d/%d: ---\n", ii, N_ix);
    X_set_2D_vec = X_set_2D(:);
    X_set_2D_vec(ix_rmv) = [];
    s2_rin = N0_rin*BW*(X_set+bias).^2;

    % Loop over OMA
    for n = 1:N_oma
        oma = db2pow(oma_dBm_vec(n))*1e-3;
        %fprintf(">> OMA = %.2f [dBm]\n", oma_dBm_vec(n));
        eta = oma/(2*(M-1));
        G = ( 10^(-0.35*L/10)*0.5*eta )^(-1);
    
        s2_th = G^2*N0_th*BW;
        s2_vec = s2_th + s2_rin;    % Noise variance vector in 1D
        
        s_vec_2D = sqrt(s2_vec) + 1j*sqrt(s2_vec');
        s_vec_2D = s_vec_2D(:);
        s_vec_2D(ix_rmv) = [];

        snr_all(ii, n) = pow2db(mean(abs(X_set_2D_vec).^2)/mean(abs(s_vec_2D).^2));
    end
end
% Get Maximum SNR at high OMA
[snr_max, ii_max] = max(snr_all(:,end));
fprintf("Maximum SNR = %.2f [dB] | Constellation Index = %d\n", snr_max, ii_max);

figure,
plot(oma_dBm_vec, snr_all(ii_max,:)); grid on;
xlabel('OMA [dBm]'); ylabel('SNR [dB]');

X_set_2D_vec = X_set_2D(:);
X_set_2D_vec(Cmb(ii_max,:)) = [];
sPlot = scatterplot(X_set_2D_vec); grid on;
sLine = findobj(sPlot, 'Type', 'Line');
set(sLine, 'MarkerSize', 16);
title('Maximum SNR');

%% (1.2) SER minimization: Analytical Upper-Bound

ser_all = zeros(N_ix, N_oma);
weights = zeros(size(Cmb,1),5);

% Loop over constellations
for ii = 1:N_ix
    ix_rmv = Cmb(ii,:);
    fprintf("--- Combination %d/%d: ---\n", ii, N_ix);
    X_set_2D_vec = X_set_2D(:);
    X_set_2D_vec(ix_rmv) = Inf;
    Xw = reshape(X_set_2D_vec, 6, 6);
    
    % Calculate weight vector
    for jj = 1:6
        d = abs( Xw(2:end,jj) - Xw(1:end-1,jj) );
        weights(ii,:) = weights(ii,:) + (abs(d(:)') < Inf);
        
        d = abs( Xw(jj,2:end) - Xw(jj,1:end-1) );
        weights(ii,:) = weights(ii,:) + (abs(d(:)') < Inf);
    end   
    s2_rin = N0_rin*BW*(X_set+bias).^2;
    
    % Loop over OMA
    for n = 1:N_oma
        oma = db2pow(oma_dBm_vec(n))*1e-3;
        %fprintf(">> OMA = %.2f [dBm]\n", oma_dBm_vec(n));
        eta = oma/(2*(M-1));
        G = ( 10^(-0.35*L/10)*0.5*eta )^(-1);

        s2_th = G^2*N0_th*BW;
        s2_vec = s2_th + s2_rin;    % Noise variance vector in 1D

        Q = qfunc( 2./( sqrt(s2_vec(1:end-1)) + sqrt(s2_vec(2:end)) ) )';
   
        ser_all(ii,n) = 2/32*weights(ii,:)*Q;
    end
end
% Get Minimum SER at high OMA
[ser_min, ii_min] = min(ser_all(:,end));
fprintf("Minimum SER = %.4E | Constellation Index = %d\n", ser_min, ii_min);

figure,
semilogy(oma_dBm_vec, ser_all(ii_min,:)); grid on;
xlabel('OMA [dBm]'); ylabel('Symbol Error Rate');

X_set_2D_vec = X_set_2D(:);
X_set_2D_vec(Cmb(ii_min,:)) = [];
sPlot = scatterplot(X_set_2D_vec); grid on;
sLine = findobj(sPlot, 'Type', 'Line');
set(sLine, 'MarkerSize', 16);
title('Minimum SER');