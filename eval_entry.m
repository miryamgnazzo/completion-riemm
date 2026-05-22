function val = eval_entry(t, param, pi0, en, fun)
%EVAL_ENTRY
% Valuta una singola entrata A(t, param{:}).
%
% INPUT
%   t     : valore del tempo
%   param : cell array, ad esempio {tau, gamma}
%   pi0   : distribuzione iniziale
%   en    : osservabile
%
% OUTPUT
%   val   : valore scalare

    nQ = length(pi0) - 1;
    Q = evalQ(nQ, param{:});

    pi0 = pi0(:);
    en  = en(:);

    %OPTION 1: exponential
   val = en' * expmv(t, Q, pi0);

    %OPTION 2: KolmogorovIntegralODE
%     W = KolmogorovIntegralODE(t, Q, pi0);
%     val = en'*W;

    %da sostituire con fun, se vogliamo cambiare
end