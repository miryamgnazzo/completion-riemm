% Script -- test con tre parametri
clear; close all; clc;

n = 5; % dimensione matrice Q = (n+1)x(n+1)

en = (0:n)';  

tau   = linspace(1.0,1.5,100); %(1)
gamma = linspace(0.05,0.15,100);
t     = linspace(0.5,3,100);

%per l'esempio (1), provo con un rango [8,8,8] oppure [9,9,9]

m = zeros(length(tau), length(gamma), length(t));

pi = zeros(1,n+1);
pi(end) = 0.5; pi(end-1) = 0.5;
%pi(end) = 1;

for i = 1:length(tau)

    for j = 1:length(gamma)

        Q = evalQ(n, tau(i), gamma(j));

        for k = 1:length(t)

            m(i,j,k) = pi * expm(t(k)*Q) * en;

        end

    end

end

A = m;
clear m

T = tensor(A);
r = zeros(1,3);

for k = 1:3
    M = double(tenmat(T,k));
    r(k) = rank(M);
end

function Q = evalQ(n, tau, gamma)
    %computing the matrix Q
    q = -(tau*ones(1,n-1) + gamma*[1:n-1]);
    q = [-tau, q, -gamma*n];
    Q = diag(tau*ones(1,n), 1) + diag(gamma*[1:n], -1) + diag(q);
end
