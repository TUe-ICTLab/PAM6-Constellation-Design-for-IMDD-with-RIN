function [G,H,LUT] = Hamming_128_120()
    % Primitive polynomial GF(2^7) : p(X) = 1 + X^3 + X^7
    % Equivalent to 1001000_1
    n = 127;
    k = 120;
    r = n - k;

    % Starting
    H = eye(r);

    for i = r+1:n
        row = H(i-1,:);
        row_ = circshift(row, 1);
        if row_(1) == 1
            H(i,:) = mod(row_ + [0 0 0 1 0 0 0], 2);
        else
            H(i,:) = row_;
        end
    end
    P = H(r+1:n,:)';

    % Structure holds H*G^T = G*H^T = 0 in GF(2)
    H = [P eye(r)];
    G = [eye(k) P'];

    % LUT for Syndrome decoding
    S_ix = bit2int(H,r)';
    LUT = zeros(n,1);
    for i = 1:n
        LUT( S_ix(i) ) = i;
    end

    % Extensions for (128, 120)
    H = [H zeros(r,1); ones(1,n+1)];
    G = [G mod(sum(G,2), 2)];

end