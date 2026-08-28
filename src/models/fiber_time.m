function v = fiber_time(tvec, param, pi0, en, fun, Qh)
%FIBER_TIME  Time fibre of the tensor at fixed parameter values.
%   Builds the generator (from QH if given, otherwise EVALQ_EXTENDED) and
%   integrates the forward Kolmogorov equations over TVEC. FUN is an
%   optional measure handle; by default the instantaneous measure is used.
    if nargin >= 6 && ~isempty(Qh)
        Q = Qh(param{:});
    else
        nQ = length(pi0) -2;
        Q = evalQ_extended(nQ, param{:}); %da migliorare
    end

    tvec = tvec(:);
    nt = length(tvec);
    v = zeros(nt, 1);

    en = en(:); %make it column

%OPTION 1:  choose the exponential
% 
% fprintf('ci arrivo? \n')
%     for k = 1 : nt
%         v(k) = en'*expmv(tvec(k), Q, pi0);
%     end

%OPTION 2: KolmogorovODE 

%QUESTO OK, ma lento se lo faccio punto per punto
% for k = 1 : nt
%     W = KolmogorovIntegralODE(tvec(k), Q, pi0);
% %     keyboard
%     v(k) = en'*W;
% end


    if nargin >= 5 && ~isempty(fun)
        % misura personalizzata via handle: fun(tvec, Q, pi0, en) -> colonna
        % (es. Under repair mediata). Il default (fun assente/vuoto) resta la
        % misura instantaneous (Reliability).
        v = fun(tvec, Q, pi0, en);
        v = v(:);
    else
        %QUESTO prova 11-06 : instantaneous, KolmogorovODE
        W = KolmogorovODE(Q, pi0, tvec);
        v = en'*W';
        v = v(:);
    end

%QUESTO OK!
%          W = KolmogorovIntegralODE(tvec, Q, pi0);
%          v = en'*W';
%          v = v(:);

end