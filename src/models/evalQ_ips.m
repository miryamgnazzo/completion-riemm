function Q = evalQ_ips(ne, L, lambda, lambda_r, lambda_b, lambda_c, ...
                       mu, mu_u, lambda_rb, lambda_rc, ...
                       lambda_r2, lambda_b2, lambda_c2)
%EVALQ_IPS  Infinitesimal generator of Case study 2 (Integrated Power Supply).
%   Two units in a 1+1 configuration fed by an external AC source. The
%   deterministic battery discharge time L is approximated by an Erlang
%   distribution with NE phases, giving a CTMC with 3 + 2*ne states.
%   Parameters follow the order of the parameter table of the paper.

nstates = 3 + 2*ne;

iU2 = 1;
iU1 = 2;
iD2 = 2 + (1:ne);
iD1 = 2 + ne + (1:ne);
iF  = nstates;

unit_fail = lambda_r + lambda_b + lambda_c + lambda_rb + lambda_rc;
erl = ne / L;

R = sparse(nstates, nstates);

% (Up, 2)
R(iU2, iD2(1)) = lambda;
R(iU2, iU1)    = 2*unit_fail;
R(iU2, iF)     = lambda_r2 + lambda_b2 + lambda_c2;

% (Up, 1)
R(iU1, iU2)    = mu_u;
R(iU1, iD1(1)) = lambda;
R(iU1, iF)     = unit_fail;

for j = 1:ne
    % (Down, 2), fase j
    R(iD2(j), iU2)    = mu;
    R(iD2(j), iD1(j)) = 2*(lambda_b + lambda_c);
    R(iD2(j), iF)     = R(iD2(j), iF) + lambda_b2 + lambda_c2;
    if j < ne
        R(iD2(j), iD2(j+1)) = erl;
    else
        R(iD2(j), iF) = R(iD2(j), iF) + erl;
    end

    % (Down, 1), fase j
    R(iD1(j), iU1) = mu;
    R(iD1(j), iF)  = R(iD1(j), iF) + lambda_b + lambda_c;
    if j < ne
        R(iD1(j), iD1(j+1)) = erl;
    else
        R(iD1(j), iF) = R(iD1(j), iF) + erl;
    end
end

Q = R - diag(R * ones(nstates, 1));
end
