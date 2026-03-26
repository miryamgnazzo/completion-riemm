% Script -- test con 4 parametri
% Markov chain as in ex 9.8, p.324 (Trivedi Bobbi),
% with an additional probability c, as in ex 10.18 p. 380

%TEST on Chebyshev points
clear; close all; clc;

n = 5; % dimensione matrice Q = (n+1)x(n+1)

en = (0:n)';  

%We choose here chebyshev points on the different intervals
tau   = chebpts(50, [1.0,1.5]); %(1)
gamma = chebpts(50, [0.05, 0.15]);
t     = chebpts(50, [0.5,3]);
c = chebpts(50, [0.1,0.99]);

m = zeros(length(tau), length(gamma), length(c), length(t));

pi = zeros(1,n+1);
pi(end) = 0.5; pi(end-1) = 0.5;
%pi(end) = 1;

for i = 1:length(tau)
  for j = 1:length(gamma)
      for q = 1:length(c)
        Q = evalQ(n, tau(i), gamma(j), c(q));
        for k = 1:length(t)
            m(i,j,q,k) = pi * expm(t(k)*Q) * en;
        end
      end
  end
end

A = m;
clear m

T = tensor(A);
r = zeros(1,4);

% for k = 1:4
%     M = double(tenmat(T,k));
%     r(k) = rank(M);
% end

%provo rank r = 8 

function Q = evalQ(n, tau, gamma, c)
    %computing the matrix Q
    Q = zeros(n+1);
    Q = Q + diag(tau*ones(n,1), 1);
    Q(2:end,1) = gamma * [1; (2:n)'*(1-c)];
    Q = Q + diag(gamma * (1:n)' * c, -1);
    Q(2,1) = gamma;
    Q = Q - diag(sum(Q,2));
end
