function [Res, Xtr] = cheb_riemm_sparse(core_dims, tensor_dims, n0, values, intervals, pi0, en, options, coefficient, F)

    if ~exist('tenrand', 'file')
        error('Tensor Toolbox richiesto');
    end

    d = length(tensor_dims);
    total_entries = prod(tensor_dims);

    %% ===== SAMPLING FIBRE =====
    nr = round(coefficient * (sum(core_dims .* tensor_dims) + prod(core_dims)));
    fprintf('target nr = %e (%2.2f%%)\n', nr, 100*nr/total_entries);

    fiber_len = tensor_dims(1);
    nfibers = ceil(nr / fiber_len);
    maxfibers = prod(tensor_dims(2:end));
    nfibers = min(nfibers, maxfibers);

    fprintf('number of fibers = %d\n', nfibers);

    fiber_ind = randperm(maxfibers, nfibers);

    subs = cell(1, d-1);
    [subs{:}] = ind2sub(tensor_dims(2:end), fiber_ind);

    %% ===== COSTRUZIONE OSSERVAZIONI SPARSE =====
    obs_subs = [];
    obs_vals = [];

    for k = 1:nfibers

        idx_param = zeros(1, d-1);
        for j = 1:d-1
            idx_param(j) = subs{j}(k);
        end

        % parametri
        param = cell(1, d-1);
        for j = 2:d
            param{j-1} = values{j}(idx_param(j-1));
        end

        % fibra temporale
        v = fiber_time(values{1}, param, pi0, en);

        % costruisci indici completi della fibra
        n1 = tensor_dims(1);
        tmp_subs = zeros(n1, d);

        tmp_subs(:,1) = (1:n1)';
        for j = 2:d
            tmp_subs(:,j) = idx_param(j-1);
        end

        obs_subs = [obs_subs; tmp_subs];
        obs_vals = [obs_vals; v];
    end

    nobs = size(obs_subs,1);
    fprintf('effective observations = %d (%2.2f%%)\n', nobs, 100*nobs/total_entries);

    %% ===== PROBLEMA RIEMANNIANO =====
    r = max(core_dims);
    problem.M = fixedranktensorembeddedfactory(tensor_dims, r*ones(1,d));

    %% ===== COST =====
    problem.cost = @(X) cost_fun(X, obs_subs, obs_vals);

    function f = cost_fun(X, subs, vals)
        pred = eval_tt_entries(X, subs);
        diff = pred - vals;
        f = 0.5 * (diff' * diff);
    end

    %% ===== EGRAD =====
    problem.egrad = @(X) egrad_fun(X, obs_subs, obs_vals, tensor_dims);

    function g = egrad_fun(X, subs, vals, sz)

        pred = eval_tt_entries(X, subs);
        diff = pred - vals;

        g = sptensor(subs, diff, sz);
    end

    %% ===== EHESS =====
    problem.ehess = @(X,eta) problem.M.tangent2ambient(X,eta);

    %% ===== PUNTO INIZIALE =====
    if ~exist('F','var') || isempty(F)
        values_start = cellfun(@(I) chebpts(n0, I), intervals, 'UniformOutput', false);
        F = eval_all(values_start, pi0, en);
    end

    [Xt,~] = cheb_approx(core_dims, tensor_dims, F, intervals);

    X0.X = Xt;

    Cpinv = cell(1,d);
    for i = 1:d
        Cpinv{i} = pinv(double(tenmat(X0.X.core, i)));
    end
    X0.Cpinv = Cpinv;

    %% ===== SOLVE =====
    Xtr = trustregions(problem, X0, options);

    %% ===== RESIDUO =====
    pred = eval_tt_entries(Xtr, obs_subs);
    Res = norm(pred - obs_vals) / norm(obs_vals);

    fprintf('relative residual = %e\n', Res);

end