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

    %Simpler CASE
    %nQ = length(pi0) - 1;
    %Q = evalQ(nQ, param{:});

    %Extended CASE
    nQ = length(pi0) -2;
    Q = evalQ_extended(nQ, param{:}); %da migliorare

    pi0 = pi0(:);
    en  = en(:);

    %OPTION 1: exponential
 %  val = en' * expmv(t, Q, pi0);

 %prova cosi
%   val = fiber_time_fast(t, param, pi0, en);
% Fast code

    %OPTION 2: KolmogorovIntegralODE
%     W = KolmogorovIntegralODE(t, Q, pi0);
%     val = en'*W;

  W = KolmogorovODE(Q, pi0, t); %?
  W = W(:);
  v = en'*W;
  v = v(:); 

    %da sostituire con fun, se vogliamo cambiare
end