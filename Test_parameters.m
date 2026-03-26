%Test with (up to) 6 parameters
% Script -- test con 6 parametri
% Markov chain as in ex 16.14, p.609 (Trivedi Bobbi),
% we fix the two probabilities p1 and p2
clear all; close all; clc;

n = 5;
en = zeros(n,1);
en(1) = 1; en(2) = 2;

p1 = 0.5; p2= 0.5;

total = 20;

%We choose here chebyshev points on the different intervals
tau   = chebpts(total, [1.0,1.5]); %(1)
gamma = chebpts(total, [0.05, 0.15]);
lambda = chebpts(total, [0.01, 0.1]);
c = chebpts(total, [0.1,0.99]);
t     = chebpts(total, [0.5,3]);

m = zeros(length(tau), length(gamma), length(lambda), length(c), length(t));

pi0 = zeros(1,n);
pi0(3) = 1;

for i = 1:length(tau)
  for j = 1:length(gamma)
      for q = 1:length(lambda)
          for p = 1:length(c)
            Q = evalQ(tau(i), gamma(j), lambda(q), c(p));
            for k = 1:length(t)
                m(i,j,q,p,k) = pi0 * expm(t(k)*Q) * en;
            end
          end
      end
  end
end

A = m;
clear m

keyboard

T = tensor(A);
r = zeros(1,5);

%provo con r = [5,5,5,5,5];

% for k = 1:5
%     M = double(tenmat(T,k));
%     s = svd
% end

%Choice of the initial tensor F
n0    = 7;
kcore = 5;

% Intervals for the cheb points
intervals = cell(1,5);
intervals{1} = [1.0, 1.5];    % tau
intervals{2} = [0.05, 0.15];  % gamma
intervals{3} = [0.01, 0.1];   % lambda
intervals{4} = [0.1, 0.99];   % c
intervals{5} = [0.5, 3.0];    % t

mfun = @(tau, gamma, lambda, c, t) local_mfun(tau, gamma, lambda, c, t, pi0, en);

% Punti iniziali di Chebyshev
tau0   = chebpts(n0, intervals{1});
gamma0 = chebpts(n0, intervals{2});
lambda0 = chebpts(n0, intervals{3});
c0 = chebpts(n0, intervals{4});
t0     = chebpts(n0, intervals{5});

%Tensore iniziale da passare a tensor_completion
[TTau, GGamma, LL, CC, TT] = ndgrid(tau0, gamma0, lambda0, c0, t0);
F = arrayfun(@(tau,gamma, lambda, c,t) mfun(tau,gamma, lambda, c,t), TTau, GGamma, LL, CC, TT);

function Q = evalQ(tau, gamma, lambda, c)
    p1 = 0.5; p2= 0.5;
    %computing the matrix Q
    Q = [-(tau + lambda),      tau,                 0,                  0,        lambda;
      gamma,              -(tau + gamma + lambda*p1), tau,          lambda*p1, 0;
      2*gamma*(1-c),      2*gamma*c,           -(2*gamma + lambda*p2), lambda*p2, 0;
      0,                  0,                   0,                  0,        0;
      0,                  0,                   0,                  0,        0];
end

function val = local_mfun(tau, gamma, lambda, c, t, pi0, en)
    Q = evalQ(tau, gamma, lambda, c);
    val = pi0 * expm(t * Q) * en;
end
