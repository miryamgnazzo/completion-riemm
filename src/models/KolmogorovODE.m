function pi = KolmogorovODE(Q, pi0, t)
%KOLMOGOROVODE  Solve the forward Kolmogorov equations.
%   Returns pi(t) = pi0 * expm(t*Q) at the times T, advancing with the
%   exact propagator over each step.
t  = t(:);
n  = length(pi0);
Qf = full(Q);
pi = zeros(length(t), n);
pi(1, :) = pi0;
for j = 2 : length(t)
    h = t(j) - t(j-1);
    pi(j, :) = pi(j-1, :) * expm(h * Qf);   % passo esatto: pi(t_{j-1}) * expm(h*Q)
end

% ---- ALTERNATIVA (Q diagonalizzabile): autodecomposizione, esatta e piu' veloce ----
% Valuta tutti i tempi in un colpo: pi(t) = (pi0*V) .* exp(t*d) * V^{-1}.
% Piu' veloce (una eig per fibra invece di una expm per passo), ma rischiosa se
% Q e' (quasi) difettoso (qui mu = mu_d = 0.5 puo' dare autovalori coincidenti).
% Decommenta per usarla al posto del ciclo qui sopra.
% t = t(:);
% [V, Dm] = eig(full(Q));  dvec = diag(Dm);
% w  = (pi0(:).') * V;
% pi = real( (w .* exp(t * dvec.')) / V );

% ---- VERSIONE PRECEDENTE (commentata, conservata): trapezi / Crank-Nicolson ----
% Pade (1,1) di expm(h*Q), 2o ordine: accurata vicino a t=0 e a regime, ma con
% errore O(h^2) nella regione di transizione.
% opts = odeset('AbsTol', 1e-12, 'RelTol', 1e-10);
% % [tt,pi2] = ode45(@(t,pi) Q'*pi, t, pi0, opts);   % ode45: accurato ma lento (stiff)
% pi = zeros(length(t), length(pi0));
% pi(1, :) = pi0;
% for j = 2 : size(pi, 1)
%     h = t(j) - t(j-1);
%     pi(j, :) = ( pi(j-1, :) + h/2 * pi(j-1,:) * Q ) / (eye(size(pi, 2)) -  h/2 * Q);
% end

end
