function v = fiber_time(tvec, param, pi0, en, fun)
%EVAL_TIME_FIBER
% Calcola la fibra temporale di un tensore A costruito come valutazione
% di una quantità dipendente dal tempo e da alcuni parametri.
%
% INPUT
%   tvec      : vettore dei tempi
%   param   : cell array con i valori fissati
%   pi0         : distribuzione iniziale (vettore riga o colonna)
%   fun         : funzione handle che valuta la misura
%
% OUTPUT
%   v           : fibra temporale, è array


    % Generatore fissato ai parametri selezionati
    % (con param che deve essere cell-array)
    
    %Simpler CASE
    %nQ = length(pi0)-1;
    %Q = evalQ(nQ, param{:});

    %Extended CASE
    nQ = length(pi0) -2;
    Q = evalQ_extended(nQ, param{:}); %da migliorare

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


%QUESTO prova 11-06
% fprintf('ci arrivo?\n')
         W = KolmogorovODE(Q, pi0, tvec);
         v = en'*W';
         v = v(:); 

%QUESTO OK!
%          W = KolmogorovIntegralODE(tvec, Q, pi0);
%          v = en'*W';
%          v = v(:); 

end