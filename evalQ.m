function Q = evalQ(n, tau, gamma, c)
%TODO: solo a due o tre parametri! da generalizzare nel caso

    if ~exist('c', 'var') || isempty(c)
    c = 1;
    end

    %computing the matrix Q
    Q = zeros(n+1);
    Q = Q + diag(tau*ones(n,1), 1);
    Q(2:end,1) = gamma * [1; (2:n)'*(1-c)];
    Q = Q + diag(gamma * (1:n)' * c, -1);
    Q(2,1) = gamma;
    Q = Q - diag(sum(Q,2));
 end

% function Q = evalQ(n, tau, gamma, c, lambda, p1, p2)
%     %p1 = 0.5; p2= 0.5;
%     %computing the matrix Q
%     Q = [-(tau + lambda),      tau,                 0,                  0,        lambda;
%       gamma,              -(tau + gamma + lambda*p1), tau,          lambda*p1, 0;
%       2*gamma*(1-c),      2*gamma*c,           -(2*gamma + lambda*p2), lambda*p2, 0;
%       0,                  0,                   0,                  0,        0;
%       0,                  0,                   0,                  0,        0];
% end