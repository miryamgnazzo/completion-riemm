%Script CASE STUDY 1: choosing nr = 3 + multilevel
clear all; close all; clc;
nreplicas = 2;
mu1 = 0.5;
mu2 = 0.5;

lambda2 = 0;
cr1 = 1;
%cr = 0.9;
c2 = 0;

%we still have
%cf, lambda

nstates = nreplicas+2; % labelled as n, n-1,..., 1, asked_rejuvenation and 0 (failed system state)

pi0 = zeros(nstates,1);
pi0(1) = 1;

en = [ones(nreplicas+1,1);0];

%Intervalli
nvariables = 4; %number of free parameters + time

tf = 24*365*10;
intervals = cell(1,nvariables);
intervals{1} = [0, tf];    % t
intervals{2} = [1.e-6, 1.e-5];    % lambda
intervals{3} = [0.9, 0.99]; % cf
intervals{4} = [0.9, 0.99]; % cf

core_dims = 4 * ones(1,nvariables);

ncheb = [4 32]; % 64 128];
coeff_levels = [1 10]; % 50];

options_levels(1).maxiter = 100;
options_levels(1).maxinner = 50;
% options_levels(1).tolcost = 1e-5;
options_levels(1).tolgradnorm = 1e-5;

options_levels(2).maxiter = 100;
options_levels(2).maxinner = 50;
% options_levels(2).tolcost = 1e-5;
options_levels(2).tolgradnorm = 1e-6;

% options_levels(3).maxiter = 100;
% options_levels(3).maxinner = 50;
% % options_levels(3).tolcost = 1e-6;
% options_levels(3).tolgradnorm = 1e-6;

% options_levels(4).maxiter = 100;
% options_levels(4).maxinner = 50;
% options_levels(4).tolcost = 1e-6;

% options_levels(4).tolgradnorm = 1e-6;

[Res, Xfinal, err, relerr] = cheb_completion_multi( ...
    core_dims, ncheb, intervals, pi0, en, ...
    options_levels, coeff_levels);