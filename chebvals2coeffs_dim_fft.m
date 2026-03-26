function A = chebvals2coeffs_dim_fft(A, dim)
% CHEBVALS2COEFFS_DIM_FFT
% Applica la trasformata valori -> coefficienti di Chebyshev
% lungo la dimensione 'dim' di un array d-dimensionale.
%
% INPUT:
%   A   : array d-dimensionale
%   dim : dimensione lungo cui applicare la trasformata
%
% OUTPUT:
%   A   : array trasformato lungo la dimensione dim

    % porta la dimensione dim in prima posizione
    perm = [dim, 1:dim-1, dim+1:ndims(A)];
    A = permute(A, perm);

    sz = size(A);
    n  = sz(1);

    % appiattisci tutte le altre dimensioni
    A = reshape(A, n, []);

    % applica la 1D colonna per colonna
    for j = 1:size(A,2)
        A(:,j) = chebvals2coeffs_fft(A(:,j));
    end

    % rimetti forma originale
    A = reshape(A, sz);
    A = ipermute(A, perm);
end