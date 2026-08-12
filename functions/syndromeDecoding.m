function [v_hat, flag] = syndromeDecoding(R, H, LUT)
    [r, n] = size(H);
    k = n-r;
    v_hat = R;
    S = mod(H*R, 2);       % Syndrome (HD)
        
    % > Error correction
    par_bit_rx = S(end,:);
    
    ix = find( (par_bit_rx==1) & (sum(S(1:end-1,:),1) > 0) );
    if ~isempty(ix)
        e = bit2int( S(1:n-k-1,ix), n-k-1 );
        bit_pos = LUT(e)';
        lin_ix = sub2ind(size(v_hat), bit_pos, ix);
    
        v_hat(lin_ix) = ~v_hat(lin_ix);
    end
    
    % > Single error at the overall parity bit (the repaired case)
    ixp = find( (par_bit_rx==1) & (sum(S(1:end-1,:),1) == 0) );
    if ~isempty(ixp)
        v_hat(n, ixp) = ~v_hat(n, ixp);
    end

    % Uncorrectable errors
    flag = find( (par_bit_rx==0) & (sum(S(1:end-1,:),1) > 0) );
end