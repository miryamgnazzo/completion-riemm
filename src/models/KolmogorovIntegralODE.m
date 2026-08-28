function W = KolmogorovIntegralODE(t, Q, pi0)
%KOLMOGOROVINTEGRALODE  Solve the integral form of the Kolmogorov equations.
%   Integrates w' = Q'*w + pi0 with w(0) = 0 using the augmented-matrix
%   (Van Loan) formulation, so every step is exact and the values do not
%   depend on the time grid. Returns one row per time when T is a vector,
%   a column vector when T is scalar.

    pi0 = pi0(:);
    t   = t(:);
    n   = length(pi0);

    M = [full(Q).', pi0; zeros(1, n+1)];   % matrice aumentata (n+1)x(n+1)

    if numel(t) == 1
        if t == 0
            W = zeros(n, 1);
            return;
        end
        z = expm(t * M) * [zeros(n,1); 1]; % un solo passo esatto 0 -> t
        W = z(1:n);
        return;
    end

    W = zeros(length(t), n);
    z = [zeros(n,1); 1];                   % w(t(1)) = 0   (t(1) = 0)
    W(1, :) = z(1:n).';
    for j = 2 : length(t)
        h = t(j) - t(j-1);
        z = expm(h * M) * z;               % passo esatto t_{j-1} -> t_j
        z(end) = 1;                        % igiene numerica (resta 1 esatto)
        W(j, :) = z(1:n).';
    end

% ---- VERSIONE PRECEDENTE (commentata): trapezi / Crank-Nicolson ----------
% Pade (1,1) on the time grid: O(h^2), and the values depend on the grid.
% DIPENDONO dalla griglia (livelli del multilivello incoerenti fra loro).
%     A  = full(Q).';                      % Q'  (convenzione a colonne)
%     In = eye(n);
%     W  = zeros(length(t), n);
%     w  = zeros(n, 1);                    % w(t(1)) = 0   (t(1) = 0)
%     W(1, :) = w.';
%     for j = 2 : length(t)
%         h = t(j) - t(j-1);
%         w = (In - (h/2)*A) \ ( (In + (h/2)*A)*w + h*pi0 );
%         W(j, :) = w.';
%     end

% ---- VERSIONE ANCORA PRECEDENTE (commentata): ode45 + interpolazione -----
%     opts = odeset('AbsTol', 1e-12, 'RelTol', 1e-10);
%     if numel(t) == 1
%         [~, sol] = ode45(@(tt,ww) Q' * ww + pi0, [0; t], zeros(size(pi0)), opts);
%         W = sol(end, :).';
%     else
%         tmax = max(t);
%         [tout, sol] = ode45(@(tt,ww) Q' * ww + pi0, [0; tmax], zeros(size(pi0)), opts);
%         W = interp1(tout, sol, t, 'linear');
%     end

end
