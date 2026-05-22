function [subs, vals] = samples(values, nsamples, pi0, en, fun)
%check on nsamples values 
    if length(values) < 2
        error('missing parameters');
    end

    tensor_dims = cellfun(@length, values);
    d = length(tensor_dims);

    N = prod(tensor_dims);
    nsamples = min(nsamples, N);

    lin_idx = randperm(N, nsamples);

    idx_cells = cell(1, d);
    [idx_cells{:}] = ind2sub(tensor_dims, lin_idx);

    subs = zeros(nsamples, d);
    for j = 1:d
        subs(:, j) = idx_cells{j}(:);
    end

    vals = zeros(nsamples, 1);

    for k = 1:nsamples
        t = values{1}(subs(k,1));

        param = cell(1, d-1);
        for j = 2:d
            param{j-1} = values{j}(subs(k,j));
        end

        vals(k) = eval_entry(t, param, pi0, en);
    end
end