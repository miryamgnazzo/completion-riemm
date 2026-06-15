function pi = KolmogorovODE(Q, pi0, t)
%KOLMOGOROVODE solve the Forward Kolmogorov system of ODEs
%   Q is the infinitesimal generator matric
%   pi0 is the initial condition, i.e., a probability vector
%   the equations is integrate from 0 to tf

opts = odeset('AbsTol', 1e-12, 'RelTol', 1e-10);

[t,pi] = ode45(@(t,pi) Q'*pi, t, pi0, opts);

end

