function Q = evalQ_extended(nreplicas, varargin)
%EVALQ_EXTENDED  Infinitesimal generator of Case study 1.
%   The number of parameters selects the model:
%     evalQ_extended(nr, lambda, cf, cr, mu_d, mu)
%         simplified model, up to 5 parameters;
%     evalQ_extended(nr, lambda, lambda2, mu, mu_d, cf, c2, c1, cr)
%         full model, with simultaneous failure of two components.
%   With lambda2 = 0 and c1 = 1 the full model reduces to the simplified one.
%
%   States: 1..nr operational components, nr+1 temporarily unavailable,
%   nr+2 permanent failure (absorbing).

np = numel(varargin);

if np >= 2 && np <= 5
    % ----- (1) modello semplificato, ordine (lambda, cf, cr, mu_d, mu) -----
    lambda = varargin{1};
    c1     = varargin{2};                                      % cf
    if np < 3 || isempty(varargin{3}), c2  = 0.9; else, c2  = varargin{3}; end % cr
    if np < 4 || isempty(varargin{4}), mu2 = 0.5; else, mu2 = varargin{4}; end % mu_d
    if np < 5 || isempty(varargin{5}), mu1 = 0.5; else, mu1 = varargin{5}; end % mu

    R = sparse(nreplicas+2,nreplicas+2);
    for i = nreplicas : -1 : 2
        % i counts the number of working replicas
        R(nreplicas-i+1, nreplicas-i+2) = i*lambda*c1;
        R(nreplicas-i+1,nreplicas+1) = i*lambda*(1-c1);
        if i<nreplicas
            R(nreplicas-i+1,nreplicas-i) = mu1;
        end
    end
    R(nreplicas, nreplicas+1) = lambda;
    R(nreplicas, nreplicas-1) = mu1;

    R(nreplicas+1,1) = c2*mu2;% rejuvenation
    R(nreplicas+1,nreplicas+2) = (1-c2)*mu2;

    Q = (R-diag(R*ones(nreplicas+2,1)));

elseif np == 8
    % ----- (2) modello completo (Fig. 3), ordine di Eq. (7) ----------------
    [lambda, lambda2, mu, mud, cf, c2, c1, cr] = varargin{:};

    nr = nreplicas;
    R = sparse(nr+2, nr+2);

    for i = nr : -1 : 2
        row = nr - i + 1;

        % guasto singolo: coperto -> i-1, scoperto -> 0
        R(row, row+1) = R(row, row+1) + cf*i*lambda;
        R(row, nr+1)  = R(row, nr+1)  + (1-cf)*i*lambda;

        % guasto simultaneo di due componenti (rate nchoosek(i,2)*lambda2):
        % coperto -> i-2 (stato 0 se i = 2), scoperto -> 0
        if i > 2
            R(row, row+2) = R(row, row+2) + c2*nchoosek(i,2)*lambda2;
        else
            R(row, nr+1)  = R(row, nr+1)  + c2*lambda2;
        end
        R(row, nr+1) = R(row, nr+1) + (1-c2)*nchoosek(i,2)*lambda2;

        % recovery di componente negli stati degradati (i < nr):
        % successo -> i+1, fallimento: resta in i (nessuna transizione)
        if i < nr
            R(row, row-1) = c1*mu;
        end
    end

    % stato 1: guasto dell'ultimo componente -> 0, recovery -> 2
    R(nr, nr+1) = lambda;
    R(nr, nr-1) = c1*mu;

    % stato 0: recovery riuscito -> nr (pienamente operativo), fallito -> 0f
    R(nr+1, 1)    = cr*mud;
    R(nr+1, nr+2) = (1-cr)*mud;

    Q = R - diag(R*ones(nr+2,1));

else
    error(['evalQ_extended: unsupported number of parameters (%d). ', ...
           'Use 2-5 (simplified model) or 8 (full model, ', ...
           'argument order: lambda, lambda2, mu, mu_d, cf, c2, c1, cr).'], np);
end
end
