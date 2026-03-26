function a = chebvals2coeffs_fft(v)
% CHEBVALS2COEFFS_FFT
% Coefficienti di Chebyshev del polinomio interpolante
% compatibili con:
%   x = chebpts(n) ordinati da -1 a 1
%   C = chebvander(x)
%
% OUTPUT:
%   p(x) = a(1) T0(x) + a(2) T1(x) + ... + a(n) T_{n-1}(x)

    v = v(:);
    n = length(v);

    if n == 1
        a = v;
        return
    end

    N = n - 1;

    % ------------------------------------------------------------
    % FFT standard vuole nodi da 1 a -1,
    % quindi qui ribaltiamo perché il tuo ordine naturale è -1 -> 1
    % ------------------------------------------------------------
    v = flipud(v);

    % ------------------------------------------------------------
    % Estensione simmetrica pari
    % ------------------------------------------------------------
    V = [v; v(N:-1:2)];

    % ------------------------------------------------------------
    % FFT
    % ------------------------------------------------------------
    F = real(fft(V));

    % ------------------------------------------------------------
    % Coefficienti Chebyshev
    % ------------------------------------------------------------
    a = F(1:n) / N;

    % Correzione estremi
    a(1)   = a(1)/2;
    a(end) = a(end)/2;
end