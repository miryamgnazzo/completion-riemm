function [vals, valsXt, err, relerr] = check_samples(Xt, values, nsamples, pi0, en, fun)
%check on random samples, using the function for extraction

    [subs, vals] = samples(values, nsamples, pi0, en);

    ns = size(subs,1);
    valsXt = zeros(ns, 1);

    for k = 1:ns
        idx = num2cell(subs(k, :));
        valsXt(k) = Xt(idx{:});
    end
    
    %keyboard

    err = abs(valsXt - vals);
    relerr = norm(err) / norm(vals);
end