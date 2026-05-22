function vals = eval_tt_entries(X, subs)
% valuta un ttensor su più indici

    n = size(subs,1);
    d = size(subs,2);

    core = X.X.core;
    U = X.X.U;

    vals = zeros(n,1);

    for k = 1:n
        v = core;
        for j = 1:d
            v = ttm(v, U{j}(subs(k,j),:), j);
        end
        vals(k) = v;
    end
end