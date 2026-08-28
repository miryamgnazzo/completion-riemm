function smoke_check()
%SMOKE_CHECK  Check that the repository is correctly installed.
%   Verifies that the shared functions resolve, that every .m file parses,
%   that the stored .mat results load and that the external dependencies
%   are on the path. Runs no experiment.

root = fileparts(mfilename('fullpath'));
setup_paths();

%% 1) shared functions
need = {'cheb_riemm_sparse','cheb_riemm_sparse_mixed','cheb_completion_multi', ...
        'cheb_approx','chebvander','chebvander_shifted','check_samples','samples', ...
        'eval_entry','eval_all','evalQ_extended','evalQ_ips','fiber_time', ...
        'KolmogorovODE','KolmogorovIntegralODE'};
bad = need(cellfun(@(f) isempty(which(f)), need));
if isempty(bad)
    fprintf('[OK]   path: all %d shared functions resolve\n', numel(need));
else
    fprintf('[FAIL] path: missing %s\n', strjoin(bad, ', '));
end

%% 2) syntax
L = dir(fullfile(root, '**', '*.m'));
nerr = 0;
for k = 1:numel(L)
    msg = checkcode(fullfile(L(k).folder, L(k).name), '-string');
    if contains(msg, 'Parse error')
        fprintf('[FAIL] parse error in %s:\n%s\n', L(k).name, msg);
        nerr = nerr + 1;
    end
end
fprintf('[%s] syntax: %d .m files checked, %d parse errors\n', ...
        ternary(nerr == 0, 'OK  ', 'FAIL'), numel(L), nerr);

%% 3) stored results
M = dir(fullfile(root, 'experiments', '**', '*.mat'));
for k = 1:numel(M)
    try
        S = load(fullfile(M(k).folder, M(k).name));
        fprintf('       %-56s %2d variables\n', M(k).name, numel(fieldnames(S)));
    catch ME
        fprintf('[FAIL] %s cannot be read: %s\n', M(k).name, ME.message);
    end
end
fprintf('[OK]   results: %d .mat files\n', numel(M));

%% 4) external dependencies, required to run the experiments
% MarkovCrossApproximation supplies the ACA comparison method and chebpts.
% It is third-party code, used unmodified, and is therefore not bundled here.
ext = {'trustregions',   'Manopt'
       'tenrand',        'Tensor Toolbox'
       'aca_nd',         'MarkovCrossApproximation (ACA)'
       'aca_eval_fiber', 'MarkovCrossApproximation (ACA)'
       'find_pivot',     'MarkovCrossApproximation (ACA)'
       'chebpts',        'MarkovCrossApproximation (Chebyshev nodes)'};
nmiss = 0;
for k = 1:size(ext,1)
    w = which(ext{k,1});
    if isempty(w)
        fprintf('[MISS] %-14s not found (%s)\n', ext{k,1}, ext{k,2});
        nmiss = nmiss + 1;
    else
        fprintf('[OK]   %-14s -> %s\n', ext{k,1}, w);
    end
end
if nmiss > 0
    fprintf(['\n%d external dependencies missing: the experiments cannot ' ...
             'run.\nSee the "Requirements" section of README.md.\n'], nmiss);
end

end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
