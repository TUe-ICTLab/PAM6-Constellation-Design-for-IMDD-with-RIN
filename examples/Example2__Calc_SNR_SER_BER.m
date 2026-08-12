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

%% (2.1) Simulation with Bit Labels: SNR, SER and pre-FEC BER
%  Discrete-time channel model
load('QAM36_idx2rmv.mat');          % Load reduced set
ix2rmv = [233 185 119];     		% [Framed-Cross=212, Cross=185, RIN=119]
N_ix = length(ix2rmv);

X_set_2D_vec_N = zeros(32,N_ix);

% --- Construct the QAM-32 sets ---
for ii = 1:N_ix
    X_set_2D_vec = X_set_2D(:);
    iixx = Cmb(ix2rmv(ii),:);
    X_set_2D_vec(iixx) = [];
    X_set_2D_vec_N(:,ii) = X_set_2D_vec;
end

% --- Bit Labels ---
bit_label_int = [0 4 5 21 20 16 1 7 23 17 3 2 6 22 18 19 11 10 14 30 26 27 9 15 31 25 8 12 13 29 28 24; ...
	             7 5 13 15 4 3 1 9 11 12 6 2 0 8 10 14 22 18 16 24 26 30 20 19 17 25 27 28 23 21 29 31; ...
	             0 1 3 2 18 16 4 5 7 6 22 20 12 13 15 14 30 28 8 9 11 10 29 24 25 27 31 26 17 19 23 21]';

% --- Variables ---
snr = zeros(N_ix, N_oma);
ser = zeros(N_ix, N_oma);
ser_ub = zeros(1, N_oma);
ber_pre = zeros(N_ix, N_oma);
Px = 1/32*ones(1,32);
bits_tx = gpuArray.randi([0 1], 5*Nsym, 1, 'single');

% --- Monte Carlo loops per GPU ---
N_MC = 10;
snr_avg = zeros(size(snr));
ser_avg = zeros(size(ser));
ber_pre_avg = zeros(size(ber_pre));

% Monte Carlo Loop
for nn = 1:N_MC
    
    fprintf("=== Monte Carlo Loop: %d/%d ===\n", nn, N_MC);
    % Constellation Loop
    for ii = 1:N_ix
        
        % Per-set variables
        [I_bk_1, I_bk_0] = LLR_Subsets(5, bit_label_int(:,ii));
        bit_label_LUT = zeros(1,32);      bit_label_LUT(bit_label_int(:,ii)+1) = 1:32;
        fprintf("> --- Combination %d/%d ---\n", ii, N_ix);
        X_set_2D_vec = X_set_2D_vec_N(:,ii);
        L_values = gpuArray.zeros(5, Nsym, 'single');
        Pxx = 1/32*ones(36,1);
        Pxx(Cmb(ix2rmv(ii),:)) = 0;
        Pxx = reshape(Pxx, 6, 6);
        Px_marginal = sum(Pxx);      % Marginalization of the probabilities (only because of symmetry)
        %s2_rin = calc_RIN_variance(pulse, rolloff, Rs, X_set, Px_marginal, bias, N0_rin, 30);
        s2_rin = N0_rin*BW*(X_set+bias).^2;

        rnd_ix = bit_label_LUT(bit2int(bits_tx, 5)+1);
        X_rnd = single( gpuArray( X_set_2D_vec( rnd_ix ) ) );
        
        % OMA Loop
        fprintf(">> OMA (dBm) = [");
        for n = 1:N_oma
            oma = db2pow(oma_dBm_vec(n))*1e-3;
            fprintf("%.1f, ", oma_dBm_vec(n));

            eta = oma/(2*(M-1));
            G = ( 10^(-0.35*L/10)*0.5*eta )^(-1);

            % Noise variance analytical model
            s2_th = G^2*(N0_th*BW);
            s2_vec_set = s2_th + s2_rin;
            s2_vec_set_2D = s2_vec_set + 1j*s2_vec_set';
            s2_vec_set_2D = s2_vec_set_2D(:);
            s2_vec_set_2D(Cmb(ix2rmv(ii),:)) = [];
            
            %Z_noise_var = Z_genie_var;
            Z_noise_var = s2_vec_set_2D;
            
            Y = X_rnd + sqrt(real(Z_noise_var(rnd_ix))).*randn(Nsym,1) + ...
                1j*sqrt(imag(Z_noise_var(rnd_ix))).*randn(Nsym,1);

            % LLRs
            for k = 1:5
                L_values(k,:) = LLR_QAM(Y, X_set_2D_vec, Px, Z_noise_var, I_bk_1(k,:), I_bk_0(k,:));
            end
            
            % HD for BER_pre and SER
            bits_rx = (L_values <= 0)*(0) + (L_values > 0)*(1);
            bits_rx = bits_rx(:);
            ber_pre(ii,n) = sum(bits_rx ~= bits_tx)/(5*Nsym);
            X_hat = single( gpuArray( X_set_2D_vec(bit_label_LUT(bit2int(bits_rx, 5)+1)) ) );
            ser(ii,n) = sum(X_rnd ~= X_hat)/Nsym;
            %ser(ii,n) = monteCarlo_QAM(Y, X_rnd, X_set_2D_vec.', 1/32*ones(1,32), real(Z_genie_var)', imag(Z_genie_var)');
    
            % SNR
            snr(ii,n) = pow2db( mean(abs(X_set_2D_vec).^2)/mean(abs(Y-X_rnd).^2) );

            % SER ub
            s2 = imag(Z_noise_var(1:6))';
            w=1/32*[24 24 20 12 12];
            Q = qfunc(2./( sqrt(s2(1:end-1)) + sqrt(s2(2:end)) ));
            s2d = 0.5*(s2(4)+s2(6));
            wd = 1/32*[2 2 4];
            Qd = [ qfunc( 2*sqrt(2)/(sqrt(s2(4))+sqrt(s2(5))) ) ...
                   qfunc( 2*sqrt(2)/(sqrt(s2(5))+sqrt(s2(6))) ) ...
                   qfunc( 2*sqrt(2)/(sqrt(s2d)+sqrt(s2(5))) )];
            ser_ub(n) = w*Q' + 0*wd*Qd';
        end
        fprintf("\b]\n");
    end
snr_avg = snr_avg + snr;
ser_avg = ser_avg + ser;
ber_pre_avg = ber_pre_avg + ber_pre;
end

% Average MC realizations
snr = snr_avg/N_MC;
ser = ser_avg/N_MC;
ber_pre = ber_pre_avg/N_MC;

% Plot figure
figure,
tt = tiledlayout(1,3);
tt.TileSpacing = 'compact';
tt.Padding = 'compact';
colororder(1/255*[69 99 168; 0 154 86; 161 33 33; 152 78 163]);

nexttile
plot(oma_dBm_vec, snr, 'LineWidth', 1.5, 'Marker','o','MarkerFaceColor','w'); grid on; hold on;
xlabel('OMA [dBm]'); ylabel('SNR [dB]');
legend('$\mathcal{X}^\mathrm{fc}$','$\mathcal{X}^\mathrm{c}$','$\mathcal{X}^\mathrm{v}$',...
        'interpreter','latex','location','southeast');

nexttile
semilogy(oma_dBm_vec, ser, 'LineWidth', 1.5,'Marker','o','MarkerFaceColor','w'); grid on; hold on;
xlabel('OMA [dBm]'); ylabel('SER');
semilogy(oma_dBm_vec, ser_ub, 'k--', 'LineWidth', 1.2);
legend('$\mathcal{X}^\mathrm{fc}$','$\mathcal{X}^\mathrm{c}$','$\mathcal{X}^\mathrm{v}$',...
        'interpreter','latex','location','northeast');

nexttile
semilogy(oma_dBm_vec, ber_pre, 'LineWidth', 1.5,'Marker','o','MarkerFaceColor','w'); grid on; hold on;
xlabel('OMA [dBm]'); ylabel('Pre-FEC BER');
semilogy(oma_dBm_vec, ser_ub/5, 'k--', 'LineWidth', 1.2);
legend('$\mathcal{X}^\mathrm{fc}$','$\mathcal{X}^\mathrm{c}$','$\mathcal{X}^\mathrm{v}$','SER UB',...
        'interpreter','latex','location','northeast');

set(gcf,'Units','Normalized','Position',[0.1 0.4, 0.8, 0.4]);