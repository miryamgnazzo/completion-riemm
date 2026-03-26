function C = chebvander(x)
    x = x(:);  %in vettore colonna 
    n = length(x);
    C = zeros(n,n);

    %first 2 columns
    C(:,1) = 1;        % first Chebyshev polynomial
    if n > 1
        C(:,2) = x;    % second Chebyshev polynomial
    end

    % ricorrenza
    for k = 2:n-1
        C(:,k+1) = 2*x.*C(:,k) - C(:,k-1);
    end
end