function pi = KolmogorovODE(Q, pi0, t)
%KOLMOGOROVODE solve the Forward Kolmogorov system of ODEs
%   Q is the infinitesimal generator matric
%   pi0 is the initial condition, i.e., a probability vector
%   the equations is integrate from 0 to tf

opts = odeset('AbsTol', 1e-12, 'RelTol', 1e-10);

% [tt,pi2] = ode45(@(t,pi) Q'*pi, t, pi0, opts);

%tic
pi = zeros(length(t), length(pi0));
pi(1, :) = pi0;

for j = 2 : size(pi, 1)
    h = t(j) - t(j-1);
    pi(j, :) = ( pi(j-1, :) + h/2 * pi(j-1,:) * Q ) / (eye(size(pi, 2)) -  h/2 * Q);
end
%toc

end

