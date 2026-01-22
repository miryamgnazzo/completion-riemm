%Script -- First test for matrix completion
clear all; close all; clc;
n = 5; %the size will be n+1
%In this example, we fix the value for tau and use gamma as variable
%tau = 1.5;
gamma = 0.01;
%t = 100;

%en = zeros(n+1,1);
%en(end) = 1;

en = 0:1:n; en = en';

%gamma = linspace(10^-5, 10^-1,500);
tau = linspace(0.5,2.5,2000);

%%gamma = linspace(10^-5, 10^-1,50);%ok per lr
%%tau = linspace(0.5,2.5,20); %ok per lr
t = linspace(0.01,10,2000);

m = zeros(length(tau),length(t));
%keyboard 

pi = zeros(1,n+1);
%pi(1) = 0.5;
%pi(2) = 0.5;

pi(end) = 0.5;
pi(end-1) = 0.5;


for i= 1: length(tau)
    Q = evalQ(n, tau(i), gamma);
    %pi = evalpi(n, tau, gamma(i));

    for j = 1: length(t)
    %create the matrix m(t,p1)
        m(i,j) = pi*expm(t(j)*Q)*en;
    end   
end

% for i= 1: length(tau)
%     for j = 1: length(gamma)
% 
%     Q = evalQ(n, tau(i), gamma(j));
%     %create the matrix m(t,p1)
%         m(i,j) = pi*expm(t*Q)*en;
%     end   
% end

%cambio nome per comodità
A = m;
clear m; clear n;


function Q = evalQ(n, tau, gamma)
    %computing the matrix Q
    q = -(tau*ones(1,n-1) + gamma*[1:n-1]);
    q = [-tau, q, -gamma*n];
    Q = diag(tau*ones(1,n), 1) + diag(gamma*[1:n], -1) + diag(q);
end

function pi = evalpi(n, tau, gamma)
    %computing the steady-state probability
    k = 0:n;
    num = (tau/gamma).^k ./ factorial(k);
    pi = num / sum(num);
end