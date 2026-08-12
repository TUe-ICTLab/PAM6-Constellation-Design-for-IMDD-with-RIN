function L = LLR_QAM(Y, X_set, Px, s2_vec, I_bk_1, I_bk_0)
    % Y:            Vector of channel observations
    % X_set:        QAM constellation (\in C)
    % Px:           Probability Distribution (\in R)
    % s2_vec:       Noise variance per symbol and per real dimension (\in C)
    % I_bk_1:       Indices where the k-th bit is 1
    % I_bk_0:       Indices where the k-th bit is 0
    
    % --- Prep dimension of signals ---
    M = length(X_set);
    X_set = X_set(:).';             % Force row vector
    Px = Px(:).';                   % Force row vector
    s2_vec = s2_vec(:).';           % Force row vector
    Y = Y(:);                       % Force column vector

    % --- Check Noise variance ---
    if isreal(s2_vec)
        s2_vec = s2_vec/2 + 1j*s2_vec/2;
    end
    if isscalar(s2_vec)
        s2_vec = repmat(s2_vec, 1, M);
    end
    
    % --- Define channel law per real dimension (Gaussian PDF) ---
    q_XgivenY = @(Y, X_set, Px, s2_vec) Px .* 1./sqrt(2*pi*s2_vec) .* exp(-(Y-X_set).^2./(2*s2_vec));
    
    % --- Get symbols subsets according to bits 1/0 ---
    Nb = size(I_bk_1,1);
    X_k_1 = zeros(Nb, M/2);
    X_k_0 = zeros(Nb, M/2);
    
    s2_k_1 = zeros(Nb, M/2);
    s2_k_0 = zeros(Nb, M/2);

    Px_k_1 = zeros(Nb, M/2);
    Px_k_0 = zeros(Nb, M/2);

    for k = 1:Nb
        X_k_1(k,:) = X_set(I_bk_1(k,:));
        X_k_0(k,:) = X_set(I_bk_0(k,:));
        s2_k_1(k,:) = s2_vec(I_bk_1(k,:));
        s2_k_0(k,:) = s2_vec(I_bk_0(k,:));
        Px_k_1(k,:) = Px(I_bk_1(k,:));
        Px_k_0(k,:) = Px(I_bk_0(k,:));
    end

    % --- Calculate LLRs ---
    L = zeros(Nb, length(Y));
    for k = 1:Nb
        % q_XgivenY_k_1 = q_XgivenY(real(Y), real(X_k_1(k,:)), Px_k_1(k,:), real(s2_k_1(k,:))) .* ...
        %                 q_XgivenY(imag(Y), imag(X_k_1(k,:)), Px_k_1(k,:), imag(s2_k_1(k,:)));
        % q_XgivenY_k_0 = q_XgivenY(real(Y), real(X_k_0(k,:)), Px_k_0(k,:), real(s2_k_0(k,:))) .* ...
        %                 q_XgivenY(imag(Y), imag(X_k_0(k,:)), Px_k_0(k,:), imag(s2_k_0(k,:)));
        % 
        % num = sum( q_XgivenY_k_1, 2 );
        % den = sum( q_XgivenY_k_0, 2 );
        
        % num = max( q_XgivenY_k_1, [], 2);
        % den = max( q_XgivenY_k_0, [], 2);
        %L(k,:) = log(num./den)';

        num = max( -0.5*log( 4*pi^2*real(s2_k_1(k,:)).*imag(s2_k_1(k,:)) ) - ( real(Y) - real(X_k_1(k,:)) ).^2./(2*real(s2_k_1(k,:)) ) - ( imag(Y) - imag(X_k_1(k,:)) ).^2./(2*imag(s2_k_1(k,:)) ) , [], 2);
        den = max( -0.5*log( 4*pi^2*real(s2_k_0(k,:)).*imag(s2_k_0(k,:)) ) - ( real(Y) - real(X_k_0(k,:)) ).^2./(2*real(s2_k_0(k,:)) ) - ( imag(Y) - imag(X_k_0(k,:)) ).^2./(2*imag(s2_k_0(k,:)) ) , [], 2);

        L(k,:) = (num - den)';
    end
end