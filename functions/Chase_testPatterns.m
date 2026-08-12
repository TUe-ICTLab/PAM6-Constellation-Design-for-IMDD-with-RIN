function xT = Chase_testPatterns(p, q, LLRs)
    [~, ix_reliable] = sort( abs(LLRs) );
    p_least_reliable = ix_reliable(1:p);
    
    NT = sum(arrayfun(@(i) nchoosek(p, i), 0:q));   % total number of test patterns
    
    % Test Patterns
    xT = zeros(length(LLRs), NT);
    flipped_bits = int2bit( (0:2^p-1), p );
    [~, ixx] = mink( sum(flipped_bits,1) , NT, 2);
    xT(p_least_reliable,:) = flipped_bits(:,ixx);
end