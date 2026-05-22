function W = KolmogorovIntegralODE(t, Q, pi0)
%KOLMOGOROVINTEGRALODE
% Risolve
%
%   w'(t) = Q' * w(t) + pi0,   w(0) = 0
%
% Se t è un vettore, restituisce una matrice W con una riga per ogni tempo.
% Se t è scalare, restituisce un vettore colonna.

    pi0 = pi0(:);
    t = t(:);

    opts = odeset('AbsTol', 1e-12, 'RelTol', 1e-10);

    if numel(t) == 1
        if t == 0
            W = zeros(size(pi0));
            return;
        end
        [~, sol] = ode45(@(tt,ww) Q' * ww + pi0, [0; t], zeros(size(pi0)), opts);
        W = sol(end, :).';
    else

        tmax = max(t);
        % integra da 0 a tmax
        [tout, sol] = ode45(@(tt,ww) Q' * ww + pi0, [0; tmax], zeros(size(pi0)), opts);
        % interpola nei punti richiesti
        W = interp1(tout, sol, t, 'linear');

        %[~, sol] = ode45(@(tt,ww) Q' * ww + pi0, t, zeros(size(pi0)), opts);
        %W = sol;
    end

   
    %keyboard


end