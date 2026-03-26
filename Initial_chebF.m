%Script for the initial approximation of the tensor 

% Dimensione matrice Q = (n+1)x(n+1)
nQ = 5;
en = (0:nQ)';

pi0 = zeros(1,nQ+1);
pi0(end)   = 0.5;
pi0(end-1) = 0.5;
% pi0(end) = 1;

% Parametri approssimazione
n0    = 20;
kcore = 10;
total = 50; %20;

% Intervalli
intervals = cell(1,4);
intervals{1} = [1.0, 1.5];    % tau
intervals{2} = [0.05, 0.15];  % gamma
intervals{3} = [0.1, 0.99];   % c
intervals{4} = [0.5, 3.0];    % t

% Funzione scalare
mfun = @(tau,gamma,c,t) local_mfun(tau, gamma, c, t, nQ, pi0, en);

% Punti iniziali di Chebyshev
tau0   = chebpts(n0, intervals{1});
gamma0 = chebpts(n0, intervals{2});
c0     = chebpts(n0, intervals{3});
t0     = chebpts(n0, intervals{4});

% Griglia 4D
[TTau, GGamma, CC, TT] = ndgrid(tau0, gamma0, c0, t0);

% Valutazione tensore iniziale
F = arrayfun(@(tau,gamma,c,t) mfun(tau,gamma,c,t), TTau, GGamma, CC, TT);

% Approssimazione tensoriale
core_dims   = kcore * ones(1,4);
tensor_dims = total * ones(1,4);

%%DEBUG
%[apprT,xx] = cheb_approx(core_dims, tensor_dims, F, intervals, mfun);
% [apprT,xx] = cheb_approx_fft(core_dims, tensor_dims, F, intervals);
% 
% %%Confronto con funzione vera
% d = 4;
% XX_points = cell(1,d);
% [XX_points{:}] = ndgrid(xx{:});
% Ftrue = arrayfun(@(tau,gamma,c,t) mfun(tau,gamma,c,t), XX_points{:});
% err = norm(full(apprT) - Ftrue);
% disp(err)


function val = local_mfun(tau, gamma, c, t, nQ, pi0, en)
    Q = evalQ(nQ, tau, gamma, c);
    val = pi0 * expm(t * Q) * en;
end


function Q = evalQ(n, tau, gamma, c)
    %computing the matrix Q
    Q = zeros(n+1);
    Q = Q + diag(tau*ones(n,1), 1);
    Q(2:end,1) = gamma * [1; (2:n)'*(1-c)];
    Q = Q + diag(gamma * (1:n)' * c, -1);
    Q(2,1) = gamma;
    Q = Q - diag(sum(Q,2));
end
