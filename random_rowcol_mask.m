function [rows, cols, M, osf] = random_rowcol_mask(m, n, r, osf)
% Select entire rows and columns uniformly at random.
% The number of observed entries is approximately
% osf * r * (m + n - r).

    % Number of rows and columns to sample
    nr = 2*ceil(osf * r * n / (m + n));
    nc = 2*ceil(osf * r * m / (m + n));

    % Sample rows and columns uniformly at random
    rows = randperm(m, min(nr, m));
    cols = randperm(n, min(nc, n));

    % Create sparse mask
    M = sparse(m, n);

    % Set sampled rows
    M(rows, :) = 1;

    % Set sampled columns
    M(:, cols) = 1;

    % Actual number of observed entries
    sample_size = nnz(M);

    % Update oversampling factor
    osf = sample_size / (r * (m + n - r));
end
