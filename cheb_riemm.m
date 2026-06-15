function [Res, Xtrfull] = cheb_riemm(core_dims, tensor_dims, n0, values, intervals, pi0, en, options, coefficient, F, fun)
    %two step optimization procedure, via chebyshev interpolation and
    %riemannian tensor completion, via tensor toolbox

    if ~exist('tenrand', 'file')
        fprintf('Tensor Toolbox version 2.6 or higher is required.\n');
        return;
    end

    %A is a multidimensional tensor in MATLAB
    %core dims contains the rank

    if length(tensor_dims)~=length(core_dims)
        fprintf('Missing rank information.\n');
        return;
    end

    total_entries = prod(tensor_dims);
    d = length(tensor_dims);


% numero target di osservazioni
nr = round(coefficient* (sum(core_dims .* tensor_dims) + prod(core_dims)));
fprintf('target nr = %e, total entries = %d (%2.1f%% of total)\n', ...
    nr, total_entries, 100 * nr / total_entries);

fiber_len = tensor_dims(1);
% numero di fibre da prendere
nfibers = ceil(nr / fiber_len);
% numero massimo di fibre possibili
maxfibers = prod(tensor_dims(2:end));
nfibers = min(nfibers, maxfibers);

fprintf('number of selected fibers = %d\n', nfibers);

% scelgo fibre casuali
fiber_ind = randperm(maxfibers, nfibers);

P = false(tensor_dims);
PA = zeros(tensor_dims);

% converto gli indici lineari in indici sulle dimensioni 2:d
subs = cell(1, d-1);
[subs{:}] = ind2sub(tensor_dims(2:end), fiber_ind);


for k = 1:nfibers
    
    idx = cell(1, d);
    idx{1} = 1:tensor_dims(1);
    
    for j = 2:d
        idx{j} = subs{j-1}(k);
    end
    
    % QUI ho che:
    % values{1} = tvec
    % values{2}, values{3}, ... = griglie dei parametri
    param = cell(1, d-1);
    for j = 2:d
        param{j-1} = values{j}(idx{j});
    end

   v = fiber_time(values{1}, param, pi0, en); %11-06
%    v = fiber_time_fast(values{1}, param, pi0, en); %11-06

    PA(idx{:}) = v;
    P(idx{:}) = true;
end
    
    % in Tensor Toolbox
    P = tensor(P);
    PA = tensor(PA);
    
    %NUMERO FINALE OSSERVAZIONI
    nobs = nnz(double(P));
    fprintf('effective observed entries = %d (%2.3f%% of total)\n', ...
        nobs, 100 * nobs / total_entries);

    %pause
    %Riemannian optimization problem

    % Pick the submanifold of tensors of size n1-by-...-by-nd of
    % multilinear rank (r1, ..., rd).
    r = max(core_dims);
    problem.M = fixedranktensorembeddedfactory(tensor_dims, r*ones(1,d));
    % Simplified version with equal ranks!
    
    problem.cost = @cost;
    function [f, store] = cost(X, store)
        if ~isfield(store, 'PXmPA')
            Xfull = full(X.X);
            store.PXmPA = P.*Xfull - PA;
        end
        f = .5*norm(store.PXmPA)^2;
    end

    % Define the Euclidean gradient of the cost function, that is, the
    % gradient of f(X) seen as a function of X without rank restrictions.
    problem.egrad =  @egrad;
    function [g, store] = egrad(X, store)
        if ~isfield(store, 'PXmPA')
            Xfull = full(X.X);
            store.PXmPA = P.*Xfull - PA;
        end
        g = store.PXmPA;
    end
    
    % Define the Euclidean Hessian of the cost at X along a vector eta.
    problem.ehess = @ehess;
    function H = ehess(X, eta)
        ambient_H = problem.M.tangent2ambient(X, eta);
        H = P.*ambient_H;
    end

    %SELECTION OF THE STARTING POINT FOR THE RIEMANNIAN OPTIMIZATION
    %n0 = 8; 
    if ~exist('F', 'var') || isempty(F)
        d_start = length(intervals);
        values_start = cell(1,d_start);
        for i = 1:d_start
            values_start{i} = chebpts(n0, intervals{i});
        end
        fprintf('F not provided, computing it ...\n');
        %pause
        F = eval_all(values_start, pi0, en); %full small tensor
    else
        fprintf('F already provided!\n');
        %pause
    end

    fprintf('Looking for chebyshev approx!\n');
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

    % Options
    %X0 = problem.M.rand();
    %TODO - da abbassare la tolerance/iterazioni, eventualmente da passare!
%     options.maxiter = 3000;
%     options.maxinner = 100;
%     options.maxtime = inf;
%     options.storedepth = 3;
%     % Target gradient norm
%     options.tolgradnorm = 1e-5;

    % Minimize the cost function using Riemannian trust-regions
%     Xtr = trustregions(problem, X0, options);

    X0 = problem.M.rand();

    Xtr = steepestdescent(problem, X0, options);

    % Display some quality metrics for the computed solution
    Xtrfull = full(Xtr.X);
    %Afull = tensor(A);
    %fprintf('||X-A||_F / ||A||_F = %g\n', norm(Xtrfull - Afull)/norm(Afull));
    fprintf('||PX-PA||_F / ||PA||_F = %g\n', norm(P.*Xtrfull - PA)/norm(PA));

    %keyboard
    Res = norm(P.*Xtrfull - PA)/norm(PA); %Relative residual on the mask
end