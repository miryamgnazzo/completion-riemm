function [Res, M] = riemann_3test(A,r)
% Given partial observation of a low rank matrix, attempts to complete it.
%
% function low_rank_matrix_completion()
%
% This example uses low-rank matrix completion to illustrate three
% different ways of optimizing under a rank constraint with Manopt.
% 
% Method 1: via (L, R) -> L*R.' parameterization (rank <= k)
%           with burermonteiroLRlift and manoptlift
% Method 2: with desingularizationfactory (rank <= k)
% Method 3: with fixedrankembeddedfactory (rank == k)
%
%
% See also: fixedrankembeddedfactory desingularizationfactory
%           manoptlift burermonteiroLRlift euclideanlargefactory

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 15, 2014
% Contributors: Bart Vandereycken
% Change log:
%   
%   Xiaowen Jiang, Aug. 20, 2021
%       Added AD to compute the egrad and the ehess  
%   NB, June 26, 2024
%       Rewrote in full to include desingularization, lifts, large matrices
    
    %% Create a problem instance
    Res = zeros(3,4);

    % We optimize over matrices of size m-by-n with rank <= r (or = r)
    [m,n] = size(A);
   % n = 1000;
    %r = rank(A)+1;    % try setting this to 11 for example

    % The structure Rmn provides functions to operate on large matrices.
    % We use this for all operations on matrices X and Xdot that exist
    % (mathematically at least) in R^(m x n). Under the hood, we avoid ever
    % creating these matrices explicitly: they are stored in one of various
    % efficient formats (sparse; factored; as functions), and this is
    % exploited for all computations.
    Rmn = euclideanlargefactory(m, n);
    
    % Select some of the m*n entries uniformly at random.
    % The dimension of the set of matrices of size m-by-n with rank r is
    % r(m+n-r). We aim to sample a multiple of that. This multiplier is
    % the oversampling factor (osf). M is sparse with 0s and 1s.
    desired_osf =5;
    [~, ~, M, actual_osf] = random_mask(m, n, r, desired_osf);
    %[rows, cols, M, actual_osf] = random_rowcol_mask(m, n, r, desired_osf);

    keyboard
    
    % Generate a ground-truth matrix of rank rtrue (potentially different
    % from r) and compute the selected entries of that matrix, in atrue.
    % rtrue = 10;
    rtrue = min(m,n); %we consider the whole matrix A

    [UU, SS, VV] =svd(A);

    Astar.U = UU(:,1:rtrue);
    Astar.V = VV(:,1:rtrue);
    Astar.S = SS(1:rtrue,1:rtrue);

    %Astar.U = stiefelfactory(m, rtrue).rand();
    %Astar.V = stiefelfactory(n, rtrue).rand();
    %Astar.S = diag(.5 + .5*rand(rtrue, 1)); % singular values of target A;
    atrue = Rmn.sparseentries(M, Astar);    % efficient sampling of entries

    %% Create a problem structure defining the problem in R^(mxn)

    % The problem structure 'downstairs' defines the problem over all mxn
    % matrices, without rank constraint and with efficient use of sparsity.
    downstairs.M = Rmn;
    downstairs.cost = @cost;
    downstairs.grad = @grad;
    downstairs.hess = @hess;
    function store = prepare(X, store)
        if ~isfield(store, 'residue')
            % Compute the possibly nonzero entries of M.*(X - A)
            store.residue = Rmn.sparseentries(M, X) - atrue;
            store = incrementcounter(store, 'sparseentries');
        end
    end
    function [f, store] = cost(X, store)
        store = prepare(X, store);
        % f(X) = .5*norm(M.*(X - A), 'fro')^2
        f = .5*norm(store.residue)^2;
    end
    function [g, store] = grad(X, store)
        store = prepare(X, store);
        % nabla f(X) = M.*(X - A)
        g = replacesparseentries(M, store.residue);
    end
    function [h, store] = hess(X, Xdot, store) %#ok<INUSD>
        % nabla^2 f(X)[Xdot] = M.*Xdot
        MXdot = Rmn.sparseentries(M, Xdot);
        h = replacesparseentries(M, MXdot);
        % Increment by 2 because Xdot can have rank up to 2r
        store = incrementcounter(store, 'sparseentries', 2);
    end
    
    % Whenever we change f, it is a good idea to check the derivatives.
    % checkgradient(downstairs);
    % checkhessian(downstairs);

    %% Define options for the manopt solvers
    options = struct();

    stats = statscounters({'sparseentries'});
    options.statsfun = statsfunhelper(stats);

    options.theta = 0.4;      % this is a parameter in trs_tCG_cached
    options.maxtime = 60; %30;     % stop if we take more than xyz seconds
    options.tolgradnorm = 1e-10;
    options.tolcost = 1e-20; %1e-12;  % stop when f(X) is close to 0
    
    %% Build an initial guess
    X0_USV.U = stiefelfactory(m, r).rand();
    X0_USV.V = stiefelfactory(n, r).rand();
    X0_USV.S = diag(rand(r, 1))/1000;

    % Cross approximation: this works well for matrices
    % J = randperm(n, 2*r); I = randperm(m, 2*r);
    % [X0_USV.U, RU] = qr(A(:, J), 0);
    % [X0_USV.V, RV] = qr(A(I, :)', 0);
    % 
    % Core = (RU / A(I,J)) * RV';
    % [UU, SS, VV] = svd(Core);
    % X0_USV.U = X0_USV.U * UU(:, 1:r);
    % X0_USV.V = X0_USV.V * VV(:, 1:r);
    % X0_USV.S = SS(1:r, 1:r);

    J = randperm(n, r); I = randperm(m, r);
    [X0_USV.U, ~] = qr(A(:, J), 0);
    [X0_USV.V, ~] = qr(A(I, :)', 0);
    X0_USV.S = diag(rand(r, 1))/1000;

    keyboard;

    X0_LR = Rmn.to_LR(X0_USV);  % the same X0 in LR format

    
    %% Lift the downtairs problem through the LR' parameterization
    lift = burermonteiroLRlift(m, n, r);
    upstairs = manoptlift(downstairs, lift);
    [X_LR, ~, LR_info] = trustregions(upstairs, X0_LR, options); %#ok<ASGLU>
   
    %keyboard
    Res(1,1) = norm(M.*(X_LR.L*X_LR.R' - A), 'fro') / norm(A .* M, 'fro');
    Res(1,2) = norm((X_LR.L*X_LR.R' - A), 'fro') / norm(A, 'fro');
    Res(1,3) = min(min(abs(X_LR.L*X_LR.R'- A))) / max(max(A));
    Res(1,4) = max(max(abs(X_LR.L*X_LR.R'- A)))/ max(max(A))
    keyboard;
 
    %% Lift the downstairs problem through the desingularization
    desing.M = desingularizationfactory(m, n, r);
    desing.cost = @cost;
    desing.egrad = @grad;
    desing.ehess = @ehess_xp;
    function [h, store] = ehess_xp(X, Xdot, store)
        % Here we compute the Euclidean Hessian.
        % The inputs are a point X and a tangent vector Xdot.
        % The latter is in tangent vector format, but we need it in ambient
        % vector format. We need to do two things here:
        %  1. Map Xdot (a tangent vector) to the ambient space,
        %     using M.tangent2ambient.
        %  2. Extract the X component of the result, because ambient
        %     vectors in the desingularization have both an X and a P
        %     component, but only the X part is relevant to us.
        Xdot_ambient = desing.M.tangent2ambient(X, Xdot).X;
        [h, store] = hess(X, Xdot_ambient, store);
    end

    [X_XP, ~, XP_info] = trustregions(desing, X0_USV, options); %#ok<ASGLU>

    keyboard
    Res(2,1) = norm(M.*(X_XP.U*X_XP.S*X_XP.V' - A), 'fro');
    Res(2,2) = norm((X_XP.U*X_XP.S*X_XP.V' - A), 'fro');
    Res(2,3) = min(min(abs(X_XP.U*X_XP.S*X_XP.V'- A)));
    Res(2,4) = max(max(abs(X_XP.U*X_XP.S*X_XP.V'- A)));


    %% Restrict the downstairs problem to matrices of rank exactly r
    
    fixedrk.M = fixedrankembeddedfactory(m, n, r);
    fixedrk.cost = @cost;
    fixedrk.egrad = @grad;
    fixedrk.ehess = @ehess_fr;
    function [h, store] = ehess_fr(X, Xdot, store)
        % Same as in ehess_xp, we must first convert Xdot
        Xdot_ambient = fixedrk.M.tangent2ambient(X, Xdot);
        [h, store] = hess(X, Xdot_ambient, store);
    end
    
    [X_FR, ~, FR_info] = trustregions(fixedrk, X0_USV, options); %#ok<ASGLU>

    keyboard
    Res(3,1) = norm(M.*(X_FR.U*X_FR.S*X_FR.V' - A), 'fro');
    Res(3,2) = norm((X_FR.U*X_FR.S*X_FR.V' - A), 'fro');
    Res(3,3) = min(min(abs(X_FR.U*X_FR.S*X_FR.V'- A)));
    Res(3,4) = max(max(abs(X_FR.U*X_FR.S*X_FR.V'- A)));


    %% Plot some statistics to compare the various approaches
    figure(1);
    clf;
    subplot(2, 1, 1);
    hplt = ...
        semilogy([LR_info.time], [LR_info.cost], '.-', ...
                 [XP_info.time], [XP_info.cost], '.-', ...
                 [FR_info.time], [FR_info.cost], '.-');
    set(hplt, 'LineWidth', 2);
    set(hplt, 'MarkerSize', 20);
    legend('LR', 'desingularisation', 'fixed rank', 'Location', 'northeast');
    xlabel('Time [s]');
    ylabel('Cost function value (training loss)');
    title(sprintf('Oversampling factor: %.4g, Observed fraction: %.4g', ...
                   actual_osf, nnz(M)/numel(M)));
    grid on;
    subplot(2, 1, 2);
    hplt = ...
            plot([LR_info.time], [LR_info.sparseentries], '.-', ...
                 [XP_info.time], [XP_info.sparseentries], '.-', ...
                 [FR_info.time], [FR_info.sparseentries], '.-');
    set(hplt, 'LineWidth', 2);
    set(hplt, 'MarkerSize', 20);
    legend('LR', 'desingularisation', 'fixed rank', 'Location', 'northeast');
    xlabel('Time [s]');
    ylabel('Equivalent-calls to sampling sparse entries of low rank matrix');
    grid on;
    
end


% This function aims to select osf*r*(m+n-r) entries uniformly at random
% out of a matrix of size m-by-n, where osf is the oversampling factor.
% The output is a sparse matrix M of size m-by-n with 
function [I, J, M, osf] = random_mask(m, n, r, osf)
    % Expected number of collisions in a random sample of b numbers among a
    % numbers with replacement.
    expected_redundant = @(a, b) ceil(b + a*(((a-1)/a)^b - 1));
    % We'll select a few more to make up for collisions.
    desired_sample_size = round(osf*r*(m+n-r));
    initial_sample_size = desired_sample_size + ...
                            4*expected_redundant(m*n, desired_sample_size);
    sample = unique(randi(m*n, initial_sample_size, 1));
    % With high probability, we have too many in our sample.
    % Trim uniformly at random until we have just the right number.
    while numel(sample) > desired_sample_size
        to_delete = unique(randi(numel(sample), ...
                           numel(sample) - desired_sample_size, 1));
        sample(to_delete) = [];
    end
    % Update the actual sample size and oversampling factor.
    sample_size = length(sample);
    osf = sample_size/(r*(m+n-r));
    % Convert the sampled linear indices into (i, j) pairs and a mask M.
    [I, J] = ind2sub([m, n], sample);
    M = sparse(I, J, ones(sample_size, 1));
end