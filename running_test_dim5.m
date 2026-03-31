%Script for running a test case, via tensor completion
%using a dimension 5 tensor (created in Test_parameters)

clear all; close all; clc;
load('tensor5.mat');
%Carica un tensore A di dimensione 5 e taglia 20 x 20 x20 x20 x20

T = tensor(A);
n = 5;
r = 5*ones(1,n);

%initial vectors for mfun
en = zeros(n,1);
en(1) = 1; en(2) = 2;
%initial probability distribution
pi0 = zeros(1,n);
pi0(3) = 1;

n0    = 7;
kcore = 5; %questo deve essere =r!

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

mfun = @(tau, gamma, lambda, c, t) local_mfun(tau, gamma, lambda, c, t, pi0, en);

%Tensore iniziale da passare a tensor_completion
[TTau, GGamma, LL, CC, TT] = ndgrid(tau0, gamma0, lambda0, c0, t0);
F = arrayfun(@(tau,gamma, lambda, c,t) mfun(tau,gamma, lambda, c,t), TTau, GGamma, LL, CC, TT);
keyboard

%Test for tensor completion
%Prima possibilità - approssimazione cross iniziale
% [Res, Xtrfull] = tensor_completion(A,r,'cross');
% 
% %Seconda possibilità - approssimazione tucker iniziale 
 [Res, Xtrfull] = tensor_completion(A,r,'tucker');
% 
% %Terza possibilità -  approssimazione via cheb interp
% [Res, Xtrfull] = tensor_completion(A,r,'cheb', F, intervals);

%Terza possibilità -  approssimazione via cheb interp
%[Res, Xtrfull] = tensor_completion(A,r,'fft', F, intervals);

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
