function val = eval_entry(t, param, pi0, en, fun)
%EVAL_ENTRY  Evaluate a single tensor entry A(t, param{:}).
%   The generator and the time integration are delegated to FIBER_TIME,
%   evaluated on a Chebyshev grid over [0, t].

    if nargin < 5, fun = []; end %#ok<NASGU>

    npts = 100;   % punti per l'integrazione su [0, t] (tarabile)

    if t == 0
        tgrid = [0; 0];                 % trivial case: initial value
    else
        tgrid = chebpts(npts, [0, t]);  % 0 = ..., ultimo = t
    end

    v   = fiber_time(tgrid, param, pi0, en, fun);   % stesso identico procedimento
    val = v(end);                                   % value at time t
end
