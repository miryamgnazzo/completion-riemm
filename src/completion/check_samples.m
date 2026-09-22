function [vals, valsXt, err, relerr] = check_samples(Xt, values, nsamples, pi0, en, fun)
%CHECK_SAMPLES  Relative error of XT against exact entries at random samples.

    if nargin < 6, fun = []; end

    [subs, vals] = samples(values, nsamples, pi0, en, fun);

    ns = size(subs,1);
    valsXt = zeros(ns, 1);

    is_tt = isa(Xt, 'ttensor');   % i ttensor non supportano l'indicizzazione ()
    dd = size(subs, 2);

    for k = 1:ns
        if is_tt
            % ttensor entry 
            cols = cell(1, dd);
            for m = 1:dd
                cols{m} = Xt.U{m}(subs(k, m), :).';
            end
            valsXt(k) = double(ttv(Xt.core, cols, 1:dd));
        else
            idx = num2cell(subs(k, :));
            valsXt(k) = Xt(idx{:});
        end
    end
    
    %keyboard

    err = abs(valsXt - vals);
    relerr = norm(err) / norm(vals);
end
