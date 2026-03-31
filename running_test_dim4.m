% Script for running a test case, via tensor completion
% using a dimension 4 tensor of size 50 x 50 x 50 x 50
% Here we load tensor A and compute only the initial tensor F

clear all; close all; clc;
load('tensor4_cheb.mat');
% Carica un tensore A di dimensione 4 e taglia 50 x 50 x 50 x 50

T = tensor(A);
r = 8*ones(1,4);

n = 5;   % dimensione matrice Q = (n+1)x(n+1)

% initial vectors for mfun
en = (0:n)';

% initial probability distribution
pi0 = zeros(1,n+1);
pi0(end)   = 0.5;
pi0(end-1) = 0.5;
% pi0(end) = 1;

n0    = 10;
kcore = 8; %deve essere uguale a r!

%n0    = 20;
%kcore = 10;

% Intervals for the cheb points
intervals = cell(1,4);
intervals{1} = [1.0, 1.5];    % tau
intervals{2} = [0.05, 0.15];  % gamma
intervals{3} = [0.1, 0.99];   % c
intervals{4} = [0.5, 3.0];    % t

% Chebyshev points of the full tensor A
tau   = chebpts(50, [1.0, 1.5]);
gamma = chebpts(50, [0.05, 0.15]);
c     = chebpts(50, [0.1, 0.99]);
t     = chebpts(50, [0.5, 3.0]);

mfun = @(tau, gamma, c, t) local_mfun(n, tau, gamma, c, t, pi0, en);

% Initial Chebyshev points
tau0   = chebpts(n0, intervals{1});
gamma0 = chebpts(n0, intervals{2});
c0     = chebpts(n0, intervals{3});
t0     = chebpts(n0, intervals{4});

% Initial tensor F to pass to tensor_completion
[TTau, GGamma, CC, TT] = ndgrid(tau0, gamma0, c0, t0);
F = arrayfun(@(tau,gamma,c,t) mfun(tau,gamma,c,t), TTau, GGamma, CC, TT);
keyboard

% Test for tensor completion
% Prima possibilità - approssimazione cross iniziale
% [Res, Xtrfull] = tensor_completion(A,r,'cross');

% Seconda possibilità - approssimazione tucker iniziale
% [Res, Xtrfull] = tensor_completion(A,r,'tucker');

% Terza possibilità - approssimazione via cheb interp
 [Res, Xtrfull] = tensor_completion(A,r,'cheb', F, intervals);

% Quarta possibilità - approssimazione via fft
% [Res, Xtrfull] = tensor_completion(A,r,'fft', F, intervals);

function Q = evalQ(n, tau, gamma, c)
    % computing the matrix Q
    Q = zeros(n+1);
    Q = Q + diag(tau * ones(n,1), 1);
    Q(2:end,1) = gamma * [1; (2:n)' * (1-c)];
    Q = Q + diag(gamma * (1:n)' * c, -1);
    Q(2,1) = gamma;
    Q = Q - diag(sum(Q,2));
end

function val = local_mfun(n, tau, gamma, c, t, pi0, en)
    Q = evalQ(n, tau, gamma, c);
    val = pi0 * expm(t * Q) * en;
end