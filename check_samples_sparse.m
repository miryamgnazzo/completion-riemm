function [vals, valsXt, err, relerr] = check_samples_sparse(Xt, values, nsamples, pi0, en, fun)
%CHECK_SAMPLES (scalable version for ttensor)
% confronta valori esatti vs approssimazione su campioni casuali

    % =========================
    % 1. sampling (your function)
    % =========================
    [subs, vals] = samples(values, nsamples, pi0, en);

    ns = size(subs,1);
    valsXt = zeros(ns,1);

    % =========================
    % 2. evaluate Xt (ttensor)
    % =========================
    core = Xt.X.core;
    U = Xt.X.U;

    d = length(U);

    for k = 1:ns

        v = core;

        % contraction mode-by-mode
        for j = 1:d
            v = ttm(v, U{j}(subs(k,j),:), j);
        end

        valsXt(k) = v(:);
        valsXt(k) = valsXt(k,1); % scalar extraction
    end

    % =========================
    % 3. error
    % =========================
    err = abs(valsXt - vals);
    relerr = norm(err) / norm(vals);
end