function [I_bk_1, I_bk_0] = LLR_Subsets(Nb, bit_labeling)
    % Nb : number of bits
    % bit_labeling : Bit Label in integer (same order as set X)

    bit_labeling = bit_labeling(:).';
    bit_set = int2bit(bit_labeling, Nb);      % Dim. Nb x M
    
    M = 2^Nb;
    I_bk_1 = zeros(Nb, M/2);
    I_bk_0 = zeros(Nb, M/2);

    for k = 1:Nb
        I_bk_1(k,:) = find(bit_set(k,:) == 1);
        I_bk_0(k,:) = find(bit_set(k,:) == 0);
    end
end