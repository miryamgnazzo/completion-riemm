function [Res, Xtrfull] = cheb_riemm_large(core_dims, tensor_dims, n0, values, intervals, pi0, en, options, F, fun)
    %two step optimization procedure, via chebyshev interpolation and
    %riemannian tensor completion, via tensor toolbox

    if ~exist('tenrand', 'file')
        fprintf('Tensor Toolbox version 2.6 or higher is required.\n');
        return;
    end

    %A is a multidimensional tensor in MATLAB
    %tensor_dims = [4, 4, 4];
    %core dims contains the rank

    if length(tensor_dims)~=length(core_dims)
        fprintf('Missing rank information.\n');
        return;
    end

    total_entries = prod(tensor_dims);
    d = length(tensor_dims);


% numero target di osservazioni
nr = round(0.5* (sum(core_dims .* tensor_dims) + prod(core_dims)));
fprintf('target nr = %e, total entries = %d (%2.1f%% of total)\n', ...
    nr, total_entries, 100 * nr / total_entries);

% Seleziono le fibre che mi servono
fiber_len = tensor_dims(1);
nfibers = ceil(nr / fiber_len);
maxfibers = prod(tensor_dims(2:end));
nfibers = min(nfibers, maxfibers);

fprintf('number of selected fibers = %d\n', nfibers);
fiber_ind = randperm(maxfibers, nfibers);


   % Convert linear indices into subscripts on modes 2:d
    sampled_mode_subs = cell(1, d-1);
    [sampled_mode_subs{:}] = ind2sub(tensor_dims(2:end), fiber_ind);

    % Collect observed entries in sparse form:
    %   obs_subs(q,:) = [i1, i2, ..., id]
    %   obs_vals(q)   = exact value at that entry
    obs_subs_cells = cell(nfibers, 1);
    obs_vals_cells = cell(nfibers, 1);

    for k = 1:nfibers

        current_idx = cell(1, d);
        current_idx{1} = 1:tensor_dims(1);

        for j = 2:d
            current_idx{j} = sampled_mode_subs{j-1}(k);
        end

        current_param = cell(1, d-1);
        for j = 2:d
            current_param{j-1} = values{j}(current_idx{j});
        end

        current_fiber = fiber_time(values{1}, current_param, pi0, en);
        current_fiber = current_fiber(:);

        fiber_subs = zeros(length(current_fiber), d);
        fiber_subs(:,1) = (1:length(current_fiber))';
        for j = 2:d
            fiber_subs(:,j) = current_idx{j};
        end

        obs_subs_cells{k} = fiber_subs;
        obs_vals_cells{k} = current_fiber;
    end

    obs_subs = vertcat(obs_subs_cells{:});
    obs_vals = vertcat(obs_vals_cells{:});

    nobs = length(obs_vals);
    fprintf('effective observed entries = %d (%2.1f%% of total)\n', ...
        nobs, 100 * nobs / total_entries);

    pause

    %Riemannian optimization problem

    % Pick the submanifold of tensors of size n1-by-...-by-nd of
    % multilinear rank (r1, ..., rd).
    r = max(core_dims);
    problem.M = fixedranktensorembeddedfactory(tensor_dims, r*ones(1,d));
    % Simplified version with equal ranks!
    
    problem.cost = @cost;
    function f = cost(X)
        Xvals = evaluate_ttensor_entries(X.X, obs_subs);
        residual = Xvals - obs_vals;
        f = 0.5 * sum(residual.^2);
    end

    % Define the Euclidean gradient of the cost function, that is, the
    % gradient of f(X) seen as a function of X without rank restrictions.
    problem.egrad =  @egrad;
%    function g = egrad(X)
%         Xvals = evaluate_ttensor_entries(X.X, obs_subs);
%         residual = Xvals - obs_vals;
% 
%         % Sparse ambient gradient supported only on observed entries
%         g = sptensor(obs_subs, residual, tensor_dims);
%     end

    function g = egrad(X)
        Xvals = evaluate_ttensor_entries(X.X, obs_subs);
        residual = Xvals - obs_vals;
    
        gdense = zeros(tensor_dims);
        for ii = 1:size(obs_subs,1)
            idxii = num2cell(obs_subs(ii,:));
            gdense(idxii{:}) = residual(ii);
        end
        g = tensor(gdense);
    end



    % Define the Euclidean Hessian of the cost at X along a vector eta.
   % problem.ehess = @ehess;
    function H = ehess(X, eta)
        ambient_H = problem.M.tangent2ambient(X, eta);
        Hvals = evaluate_general_tensor_entries(ambient_H, obs_subs);

        % Hessian action restricted to observed entries
        H = sptensor(obs_subs, Hvals, tensor_dims);
    end

    %SELECTION OF THE STARTING POINT FOR THE RIEMANNIAN OPTIMIZATION
    if ~exist('F', 'var') || isempty(F)
        d_start = length(intervals);
        values_start = cell(1,d_start);
        for i = 1:d_start
            values_start{i} = chebpts(n0, intervals{i});
        end
        fprintf('F not provided, computing it ...\n');
        pause
        F = eval_all(values_start, pi0, en); %full small tensor
    else
        fprintf('F already provided!\n');
        pause
    end

    [Xt, ~] = cheb_approx(core_dims, tensor_dims, F, intervals);
    
    if isempty(Xt)
       error('initial point not assigned')
    end

    X0.X = Xt;
    Cpinv = cell(1, length(tensor_dims));
    for i = 1:length(tensor_dims)
       Cpinv{i} = pinv(double(tenmat(X0.X.core, i)));
    end
    X0.Cpinv = Cpinv;


    % Minimize the cost function using Riemannian trust-regions
    Xtr = trustregions(problem, X0, options);

    approx_obs_vals = evaluate_ttensor_entries(Xtr.X, obs_subs);
    Res = norm(approx_obs_vals - obs_vals) / max(norm(obs_vals), eps);
    fprintf('||PX-PA||_F / ||PA||_F = %g\n', Res);

    %Final outpot
    Xtrfull = Xtr.X;

%due funzioni aggiuntive

function vals = evaluate_ttensor_entries(T, subs_mat)
% Evaluate a Tucker tensor T only at entries listed in subs_mat.
%
% T is expected to be a ttensor with fields:
%   T.core
%   T.U{1}, ..., T.U{d}
%
% subs_mat is ns x d.

    ns = size(subs_mat, 1);
    dloc = size(subs_mat, 2);
    vals = zeros(ns, 1);

    core_array = double(T.core);
    rank_vec = size(core_array);

    if isscalar(rank_vec) && dloc > 1
        rank_vec = [rank_vec, ones(1, dloc-1)];
    end

    core_lin_total = prod(rank_vec);

    for q = 1:ns
        row_factors = cell(1, dloc);
        for mode = 1:dloc
            row_factors{mode} = T.U{mode}(subs_mat(q, mode), :);
        end

        acc = 0;

        for alpha_lin = 1:core_lin_total
            alpha = cell(1, dloc);
            [alpha{:}] = ind2sub(rank_vec, alpha_lin);

            term = core_array(alpha{:});
            for mode = 1:dloc
                term = term * row_factors{mode}(alpha{mode});
            end
            acc = acc + term;
        end

        vals(q) = acc;
    end
end

function vals = evaluate_general_tensor_entries(T, subs_mat)
% Evaluate entries of a generic tensor object T at subs_mat.
% Works with numeric arrays, tensor, sptensor, and ttensor.

    if isa(T, 'ttensor')
        vals = evaluate_ttensor_entries(T, subs_mat);
        return;
    end

    ns = size(subs_mat, 1);
    vals = zeros(ns, 1);

    for q = 1:ns
        idxq = num2cell(subs_mat(q, :));
        vals(q) = T(idxq{:});
    end
end

end