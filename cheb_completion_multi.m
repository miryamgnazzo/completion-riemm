function [Res, Xfinal, max_err, max_relerr] = cheb_completion_multi( ...
    core_dims, ncheb, intervals, pi0, en, options_levels, coeff_levels)

% MULTI-LEVEL Chebyshev + Riemannian tensor completion
%
% ncheb         = vector [n1 n2 ... nL]
% options_levels = struct array per ogni livello
% coeff_levels   = coefficienti per cheb_riemm

    if ~exist('tenrand', 'file')
        error('Tensor Toolbox version 2.6 or higher is required.');
    end

    d = length(intervals);
    nlevels = length(ncheb);

    if length(coeff_levels) ~= nlevels
        error('coeff_levels must match number of levels');
    end

    % initial tensor
    Xprev = [];

    Res = zeros(1, nlevels);
    max_err = zeros(1, nlevels);
    max_relerr = zeros(1, nlevels);

    for lev = 1:nlevels

        fprintf('\n===== LEVEL %d / %d =====\n', lev, nlevels);

        n = ncheb(lev);

        % Chebyshev nodes
        values = cell(1,d);
        for i = 1:d
            values{i} = chebpts(n, intervals{i});
        end

        % options for this level
        opts = options_levels(lev);

        coeff = coeff_levels(lev);

        % run completion
        if isempty(Xprev)
            [Res(lev), X] = cheb_riemm( ...
                core_dims, n*ones(1,d), ...
                n, values, intervals, ...
                pi0, en, opts, coeff, true);
        else
            [Res(lev), X] = cheb_riemm( ...
                core_dims, n*ones(1,d), ...
                ncheb(max(lev-1,1)), values, intervals, ...
                pi0, en, opts, coeff, false, Xprev);
        end

        % error check
%        [~, ~, err, relerr] = check_samples(X, values, 500, pi0, en);

%        max_err(lev) = max(err);
%        max_relerr(lev) = max(relerr);

 %       fprintf('Level %d error: abs = %.3e, rel = %.3e\n', ...
 %           lev, max_err(lev), max_relerr(lev));

        % update
        Xprev = X;

    end

    Xfinal = Xprev;

end