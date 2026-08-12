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

%% (6.1) Simulation of post-FEC BER ext. Hamming Code with GS constellation
if isempty(gcp('nocreate'))
    parpool(6);
end

ix2rmv = [233 119 185];% 313];		% [Framed-Cross ECOC Cross maxSNR]
N_ix = length(ix2rmv);

X_set_2D_vec_N = zeros(32,N_ix);

% --- Choose constellation ---
ix = 1;
load_string = sprintf("PAM6_BER_GS_FEC_%dG_constellation%d.mat",Rb*1e-9,ix);
load(load_string);
Cmb_ix_rmv = Cmb(ix2rmv(ix),:);
%X_opt_gs = X_opt_gs(16:end,:);

% --- Bit Labels ---
bit_label_int = [0 4 5 21 20 16 1 7 23 17 3 2 6 22 18 19 11 10 14 30 26 27 9 15 31 25 8 12 13 29 28 24; ...
                 0 1 3 2 18 16 4 5 7 6 22 20 12 13 15 14 30 28 8 9 11 10 29 24 25 27 31 26 17 19 23 21; ...
                 %0 1 3 2 18 16 4 5 7 6 22 20 12 13 15 14 30 28 8 9 11 10 29 24 25 27 31 21 17 19 23 26; ...
	             7 5 13 15 4 3 1 9 11 12 6 2 0 8 10 14 22 18 16 24 26 30 20 19 17 25 27 28 23 21 29 31]';
[I_bk_1, I_bk_0] = LLR_Subsets(5, bit_label_int(:,ix));
bit_label_LUT = zeros(1,32);      bit_label_LUT(bit_label_int(:,ix)+1) = 1:32;


% --- Variables ---
ber_pre = zeros(1, N_oma);
ber_post_HD = zeros(1, N_oma);
ber_post_SD = zeros(1, N_oma);
Px = 1/32*ones(1,32);

N_codewords = 2e6;
Nsym_RS = 12*N_codewords;       % 12 x N for Hamming payload
Nbits = 10*Nsym_RS;             % 10 bits per RS symbol

% --- Encoding ---
% Hamming code parameters
N = 128;
K = 120;
% Chase decoding parameters
p = 3;  q = 2;

bits_tx = gpuArray.randi([0 1], Nbits, 1, 'single');
u = reshape(bits_tx, K, Nbits/K);       % columns = codewords
[G, H, LUT] = Hamming_128_120();
v = mod(G'*u, 2);
bits_to_mod = v(:);
Nbits_preFEC = length(bits_to_mod);
bits_to_mod = bit2int(bits_to_mod, 5);
rnd_ix = bit_label_LUT(bits_to_mod+1);

% --- Monte Carlo loops per GPU ---
N_MC = 10;
ber_pre_avg = zeros(size(ber_pre));
ber_post_HD_avg = zeros(size(ber_post_HD));
ber_post_SD_avg = zeros(size(ber_post_SD));

% Monte Carlo Loop
for nn = 1:N_MC
    
    fprintf("=== Monte Carlo Loop: %d/%d ===\n", nn, N_MC);
            
    % Per-set variables
    
    % OMA Loop
    fprintf(">> OMA (dBm) = [");
    for ii = 1:N_oma
        oma = db2pow(oma_dBm_vec(ii))*1e-3;
        fprintf("%.1f, ", oma_dBm_vec(ii));
        
        s2_rin = N0_rin*BW*( X_opt_gs(ii,:)+bias ).^2;
        X_set_2D = X_opt_gs(ii,:) + 1j*X_opt_gs(ii,:)';
        X_set_2D_vec = X_set_2D(:);
        X_set_2D_vec(Cmb_ix_rmv) = [];
        X_rnd = single( gpuArray( X_set_2D_vec( rnd_ix ) ) );
        Nsym = length(X_rnd);
        L_values = gpuArray.zeros(5, Nsym, 'single');
        
        eta = oma/(2*(M-1));
        G = ( 10^(-0.35*L/10)*0.5*eta )^(-1);

        % Noise variance analytical model
        s2_th = G^2*(N0_th*BW);
        s2_vec_set = s2_th + s2_rin;
        s2_vec_set_2D = s2_vec_set + 1j*s2_vec_set';
        s2_vec_set_2D = s2_vec_set_2D(:);
        s2_vec_set_2D(Cmb(ix2rmv(ix),:)) = [];
        
        %Z_noise_var = Z_genie_var;
        Z_noise_var = s2_vec_set_2D;
        
        Y = X_rnd + sqrt(real(Z_noise_var(rnd_ix))).*randn(Nsym,1) + ...
            1j*sqrt(imag(Z_noise_var(rnd_ix))).*randn(Nsym,1);

        % LLRs
        for k = 1:5
            L_values(k,:) = LLR_QAM(Y, X_set_2D_vec, Px, Z_noise_var, I_bk_1(k,:), I_bk_0(k,:));
        end
        
        % HD for BER_pre 
        bits_rx = (L_values <= 0)*(0) + (L_values > 0)*(1);
        r = reshape(bits_rx(:), size(v));
        ber_pre(ii) = sum(r ~= v,'all')/(Nbits_preFEC);

        % HD decoding w/ Syndrome
        [v_hat, ~] = syndromeDecoding(r, H, LUT);
        u_decoded = v_hat(1:K,:);
        ber_post_HD(ii) = sum(u_decoded ~= u, 'all')/(Nbits);
        clear v_hat;

        % SD decoding with Chase
        r_hat = gather(r);
        L_vals = gather(L_values);
        L_vals = reshape(L_vals(:), size(v));
        parfor (jj = 1:N_codewords, 12)
            codeword = r_hat(:,jj);
            curr_LLRs = L_vals(:,jj);
    
            xT = Chase_testPatterns(p, q, curr_LLRs);
            z = mod(codeword + xT, 2);
    
            [v_hat_chase, decoding_flag] = syndromeDecoding(z, H, LUT);
            W = vecnorm( curr_LLRs - (2*v_hat_chase-1), 2).^2;     % Euclidian distance
            if ~isempty(decoding_flag)
                W(decoding_flag) = Inf;
            end
            [~,ixx] = min(W);
    
            r_hat(:,jj) = v_hat_chase(:,ixx);
        end
        r_decoded = r_hat(1:K,:);
        ber_post_SD(ii) = sum(r_decoded ~= u, 'all')/Nbits;

        
    end
    fprintf("\b]\n");
    ber_pre_avg = ber_pre_avg + ber_pre;
    ber_post_HD_avg = ber_post_HD_avg + ber_post_HD;
    ber_post_SD_avg = ber_post_SD_avg + ber_post_SD;
end

% Average MC realizations
ber_pre = ber_pre_avg/N_MC;
ber_post_HD = ber_post_HD_avg/N_MC;
ber_post_SD = ber_post_SD_avg/N_MC;

% Plot figure
figure,
colororder(1/255*[69 99 168; 0 154 86; 161 33 33; 152 78 163]);

semilogy(oma_dBm_vec, ber_pre, 'LineWidth', 1.5, 'Marker','o','MarkerFaceColor','w'); grid on; hold on;
semilogy(oma_dBm_vec, ber_post_HD, 'LineWidth', 1.5, 'Marker','o','MarkerFaceColor','w');
semilogy(oma_dBm_vec, ber_post_SD, 'LineWidth', 1.5, 'Marker','o','MarkerFaceColor','w');
xlabel('OMA [dBm]'); ylabel('Bit Error Rate');

set(gcf,'Units','Normalized','Position',[0.1 0.1, 0.6, 0.6]);

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