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

% Indices for symbol removal in QAM-36 -> QAM-32
% load('QAM36_idx2rmv.mat');          % Reduced set (ECOC 2025)
load('QAM36_idx2rmv_ext.mat');      % Full set (JSTQE 2026)
N_ix = length(Cmb);

%% (5.1) Geometric shaping optimization with BER
%  Discrete-time channel model
if isempty(gcp('nocreate'))
    parpool(6);
end

ix2rmv = [233 119];% 313];		% [Framed-Cross RIN]
N_ix = length(ix2rmv);

% --- Bit Labels ---
bit_label_int = [0 4 5 21 20 16 1 7 23 17 3 2 6 22 18 19 11 10 14 30 26 27 9 15 31 25 8 12 13 29 28 24; ...
	             0 1 3 2 18 16 4 5 7 6 22 20 12 13 15 14 30 28 8 9 11 10 29 24 25 27 31 26 17 19 23 21]';

ix = 2;
Cmb_ix_rmv = Cmb(ix2rmv(ix),:);

% --- Variables ---
snr = zeros(N_ix, N_oma);
ser = zeros(N_ix, N_oma);
ber_gs = zeros(N_oma, 1);
ber_es = zeros(N_oma, 1);
X_opt_gs = zeros(N_oma, M);
W = 2/32*[12 12 10 6 6];        % Weights RIN-constellation
%W = 2/32*[8 8 12 8 8];         % Weights Framed-Cross
Px = 1/32*ones(1,32);
Nsym = 1e4;
bits_tx = gpuArray.randi([0 1], 5*Nsym, 1, 'single');

% --- Optimization Variables ---
A = [1 -1 0 0 0 0; 
     0 1 -1 0 0 0; 
     0 0 1 -1 0 0; 
     0 0 0 1 -1 0; 
     0 0 0 0 1 -1];

b = 1e-3*ones(size(A,1),1);
Aeq = zeros(M,M);   Aeq(1,1) = 1;   Aeq(M,M) = 1;
beq = zeros(M,1);   beq(1) = -(M-1);    beq(end) = M-1;

lb = -(M-1)*ones(M,1);
ub = (M-1)*ones(M,1);

optionsA = optimoptions('fmincon');
optionsA.Algorithm = 'sqp';
optionsA.MaxFunctionEvaluations = 10000; % Adjust the value as needed
optionsA.MaxIterations = 10000; % Adjust the value as needed
optionsA.StepTolerance = 1e-15; % Adjust the value as needed
optionsA.ConstraintTolerance = 1e-15; % Adjust the value as needed
optionsA.Display = 'none';

optionsB = optimoptions('patternsearch', ...
    'UseParallel', true, ...
    'PollMethod', 'GSSPositiveBasis2N', ...
    'MeshTolerance',1e-2,...
    'Display', 'none');

% --- Monte Carlo loops per GPU ---
N_MC = 10;
%snr_avg = zeros(size(snr));
%ser_avg = zeros(size(ser));
ber_gs_avg = zeros(size(ber_gs));
ber_es_avg = zeros(size(ber_es));
X_opt_gs_avg = zeros(size(X_opt_gs));

% Monte Carlo Loop
for nn = 1:N_MC
    
    fprintf("=== Monte Carlo Loop: %d/%d ===\n", nn, N_MC);
        
    % Per-set variables
    [I_bk_1, I_bk_0] = LLR_Subsets(5, bit_label_int(:,ix));
    bit_label_LUT = zeros(1,32);      bit_label_LUT(bit_label_int(:,ix)+1) = 1:32;
    %fprintf("> --- Combination %d/%d ---\n", ix, N_ix);
    
    % OMA Loop
    fprintf(">> OMA (dBm) = [");
    for n = 1:N_oma
        oma = db2pow(oma_dBm_vec(n))*1e-3;
        delta = oma/10;
        X_set6 = -(M-1):2:(M-1);
        fprintf("%.1f, ", oma_dBm_vec(n));
        
        % Noise variance analytical model
        eta = oma/(2*(M-1));
        G = ( 10^(-0.35*L/10)*0.5*eta )^(-1);
        s2_th = G^2*(N0_th*BW);
        
        % First optimization
        if n > 1
            x0 = X_opt_gs(n-1,:); %+0.1*randn(1,M);
            %x0 = delta*X_set;
            %x0 = x0 - 0.5;
        else
            x0 = X_set6;
        end
        ber_es(n) = MonteCarloBER(bits_tx, bit_label_LUT, I_bk_1, I_bk_0, X_set6, bias, s2_th, N0_rin*BW, Cmb_ix_rmv);

        X_gs = fmincon(@(x) ser_ub_analytical( x, bias, W, s2_th, N0_rin*BW ), x0', A, b, Aeq, beq, lb, ub, [], optionsA);
        %X_gs=X_set6;
        
        [X_gs_2, ber] = patternsearch(@(x) MonteCarloBER(bits_tx, bit_label_LUT, I_bk_1, I_bk_0, x, bias, s2_th, N0_rin*BW, Cmb_ix_rmv), ...
                       X_gs, A, b, Aeq, beq, lb, ub, [], optionsB);

        ber_gs(n) = ber;
        X_opt_gs(n,:) = X_gs_2;
        %ber = MonteCarloBER(bits_tx, bit_label_LUT, I_bk_1, I_bk_0, x, bias, s2_th, N0_rin*BW, Cmb_ix_rmv);

    end
    fprintf("\b]\n");

%snr_avg = snr_avg + snr;
%ser_avg = ser_avg + ser;
%ber_pre_avg = ber_pre_avg + ber_pre;
ber_gs_avg = ber_gs_avg + ber_gs;
ber_es_avg = ber_es_avg + ber_es;
X_opt_gs_avg = X_opt_gs_avg + X_opt_gs;
end

% Average MC realizations
% snr = snr_avg/N_MC;
% ser = ser_avg/N_MC;
ber_gs = ber_gs_avg/N_MC;
ber_es = ber_es_avg/N_MC;
X_opt_gs = X_opt_gs_avg/N_MC;

% Plot figure
figure,
tt = tiledlayout(1,2);
tt.TileSpacing = 'compact';
tt.Padding = 'compact';
colororder(1/255*[69 99 168; 0 154 86; 161 33 33; 152 78 163]);

nexttile
semilogy(oma_dBm_vec, ber_es, 'LineWidth', 1.5, 'Marker','o','MarkerFaceColor','w'); grid on; hold on;
semilogy(oma_dBm_vec, ber_gs, 'LineWidth', 1.5, 'Marker','o','MarkerFaceColor','w'); grid on; hold on;
xlabel('OMA [dBm]'); ylabel('BER');

nexttile
plot(oma_dBm_vec, X_opt_gs, 'LineWidth', 1.5, 'Marker','o','MarkerFaceColor','w'); grid on; hold on;
xlabel('OMA [dBm]'); ylabel('Normalized X');
% nexttile
% semilogy(oma_dBm_vec, ser, 'LineWidth', 1.5,'Marker','o','MarkerFaceColor','w'); grid on; hold on;
% xlabel('OMA [dBm]'); ylabel('SER');
% semilogy(oma_dBm_vec, ser_ub, 'k--', 'LineWidth', 1.2);
% 
% nexttile
% semilogy(oma_dBm_vec, ber_pre, 'LineWidth', 1.5,'Marker','o','MarkerFaceColor','w'); grid on; hold on;
% xlabel('OMA [dBm]'); ylabel('Pre-FEC BER');
% semilogy(oma_dBm_vec, ser_ub/5, 'k--', 'LineWidth', 1.2);

set(gcf,'Units','Normalized','Position',[0.1 0.4, 0.8, 0.4]);

save_string = sprintf("PAM6_BER_GS_FEC_%dG_constellation%d.mat",Rb*1e-9,ix);
save(save_string, 'X_opt_gs');

%% Aux function
function ber = MonteCarloBER(bits_tx, bit_label_LUT, I_bk_1, I_bk_0, X_set6, bias, s2_th, N0_rin_BW, Cmb_ix_rmv)

    X_set_2D = X_set6 + 1j*X_set6';
    X_set_2D_vec = X_set_2D(:);
    X_set_2D_vec(Cmb_ix_rmv) = [];
    
    rnd_ix = bit_label_LUT(bit2int(bits_tx, 5)+1);
    X_rnd = single( gpuArray( X_set_2D_vec( rnd_ix ) ) );
    Nsym = length(X_rnd);
    L_values = gpuArray.zeros(5, Nsym, 'single');

    s2_rin = N0_rin_BW*(X_set6+bias).^2;

    s2_vec_set = s2_th + s2_rin;
    s2_vec_set_2D = s2_vec_set + 1j*s2_vec_set';
    s2_vec_set_2D = s2_vec_set_2D(:);
    s2_vec_set_2D(Cmb_ix_rmv) = [];
    
    %Z_noise_var = Z_genie_var;
    Z_noise_var = s2_vec_set_2D;
    
    Y = X_rnd + sqrt(real(Z_noise_var(rnd_ix))).*randn(Nsym,1) + ...
        1j*sqrt(imag(Z_noise_var(rnd_ix))).*randn(Nsym,1);

    % LLRs
    Px = 1/32*ones(32,1);
    for k = 1:5
        L_values(k,:) = LLR_QAM(Y, X_set_2D_vec, Px, Z_noise_var, I_bk_1(k,:), I_bk_0(k,:));
    end
    
    % HD for BER_pre and SER
    bits_rx = (L_values <= 0)*(0) + (L_values > 0)*(1);
    bits_rx = bits_rx(:);
    ber = sum(bits_rx ~= bits_tx)/(5*Nsym);
end

function ser = ser_ub_analytical(X, bias, w, s2_th, s2_rin)
    deltaX = X(2:end) - X(1:end-1);
    s2_vec = s2_th + (X+bias).^2*s2_rin;
    s_vec = sqrt(s2_vec(1:end-1))+sqrt(s2_vec(2:end));

    Q_vec = qfunc( deltaX./s_vec );
    ser = w(:)'*Q_vec(:);
end