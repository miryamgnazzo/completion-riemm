function Script_case1new_cheb_dsweep_ordEX1(cfg)
%SCRIPT_CASE1NEW_CHEB_DSWEEP_ORDEX1  Case study 1, reliability, d = 6..9.
%   Accuracy and cost of Chebyshev + Riemannian completion as the number of
%   parameters grows, on a grid of 256 points per mode. Parameters are
%   activated in the order of the parameter table of the paper (PAR_ORDER).
%
%   Writes case1new_cheb_dsweep_ordEX1_results.mat.

if nargin < 1, cfg = struct(); end
g = @(f,v) getdef(cfg,f,v);

here = fileparts(mfilename('fullpath'));
% Le funzioni condivise stanno in <repo>/src, messe sul path da setup_paths.m
%  (gli script vivono in <repo>/experiments/<esperimento>/).
repoRoot = fileparts(fileparts(here));
run(fullfile(repoRoot, 'setup_paths.m'));
if ~exist('chebpts','file'),      error('chebpts not found: add MarkovCrossApproximation to the path.'); end
if ~exist('tenrand','file'),      error('Tensor Toolbox is required on the path.'); end
if ~exist('trustregions','file'), error('Manopt is required on the path.'); end

%% ---- CONFIG --------------------------------------------------------------
d_list       = g('d_list', [6 7 8 9]);
ncheb        = g('ncheb', [3 16 32 64 128 256]);
coeff_levels = g('coeff_levels', [1 5 10 30 50 80]);
core_rank    = g('core_rank', 3);
nsamples     = g('nsamples', 2000);

maxiter      = g('maxiter', 50);
maxinner     = g('maxinner', 20);
tolgradnorm  = g('tolgradnorm', 1e-9);

run_seed     = g('run_seed', 1);
sample_seed  = g('sample_seed', 0);
resume       = g('resume', true);

% ordine di attivazione = tab:case1_parameters (par_2..par_9)
par_order  = g('par_order', [1 5 8 4 3 2 7 6]);
allpar_tab = g('allpar_tab', { [1e-8,1e-5], [0.9,0.99], [0.9,0.99], [0.25,0.75], ...
                               [0.25,0.75], [1e-7,1e-6], [0.95,0.995], [0.8,0.95] });
% NB: par_2 = lambda ha l'intervallo ALLARGATO a [1e-8,1e-5], come dichiarato
% nel testo ("we widen the interval chosen for parameter par_2").

outfile = g('outfile', 'case1new_cheb_dsweep_ordEX1_results.mat');
logfile = g('logfile', 'case1new_cheb_dsweep_ordEX1_progress.txt');
if ~isAbsPath(outfile), outfile = fullfile(here, outfile); end
if ~isAbsPath(logfile), logfile = fullfile(here, logfile); end

nlevels = numel(ncheb);
if numel(coeff_levels) ~= nlevels
    error('coeff_levels must have the same length as ncheb');
end
if numel(par_order) ~= 8 || ~isequal(sort(par_order), 1:8)
    error('par_order must be a permutation of 1..8');
end

if resume && isfile(outfile), lmode = 'a'; else, lmode = 'w'; end
fid=fopen(logfile,lmode); fprintf(fid,'start %s\n', datestr(now)); fclose(fid);

%% ---- Modello CONDIVISO (Case study 1 completo, nr = 5) -------------------
nreplicas = 5;
nstates   = nreplicas + 2;             % = 7
tf = 24*365*10;
pi0 = zeros(nstates,1); pi0(1) = 1;
r   = [ones(nreplicas+1,1); 0];        % reward Reliability

appendlog(logfile, sprintf(['EX2 dsweep ORDINE tab:case1_parameters | nr=%d ' ...
    'nstates=%d t_max=%d h | ncheb=[%s] coeff=[%s] rango=%d | run_seed=%d ' ...
    'sample_seed=%d nsamples=%d'], nreplicas, nstates, tf, num2str(ncheb), ...
    num2str(coeff_levels), core_rank, run_seed, sample_seed, nsamples));
appendlog(logfile, sprintf('par_order = [%s]  (posizione in evalQ_extended)', ...
    num2str(par_order)));

nd = numel(d_list);
time_cr  = nan(1, nd);
err_L2   = nan(1, nd);
err_Linf = nan(1, nd);
obs_last = nan(1, nd);
obs_tot  = nan(1, nd);
tot_entr = nan(1, nd);
done_d   = false(1, nd);

if resume && isfile(outfile)
    S = load(outfile);
    ok = isequal(S.d_list(:).', d_list(:).') && isequal(S.ncheb(:).', ncheb(:).') && ...
         isequal(S.coeff_levels(:).', coeff_levels(:).') && S.core_rank == core_rank && ...
         S.nsamples == nsamples && S.run_seed == run_seed && ...
         S.sample_seed == sample_seed && isequal(S.par_order(:).', par_order(:).');
    if ~ok
        error('resume=true ma %s ha una cfg diversa. Cancellalo o cambia outfile.', outfile);
    end
    time_cr=S.time_cr; err_L2=S.err_L2; err_Linf=S.err_Linf;
    obs_last=S.obs_last; obs_tot=S.obs_tot; tot_entr=S.tot_entr; done_d=S.done_d;
    appendlog(logfile, sprintf('RESUME: %d/%d valori di d gia'' fatti.', nnz(done_d), nd));
end

for id = 1:nd
    d = d_list(id);
    if done_d(id)
        appendlog(logfile, sprintf('d=%d gia'' fatto (err=%.3e, %.0f s), salto', ...
            d, err_L2(id), time_cr(id)));
        continue
    end
    p = d - 1;
    if p < 5 || p > 8
        error('d=%d out of range: 5..8 free parameters required (d=6..9).', d);
    end

    % liberi = primi p nell'ordine di TABELLA; fissi = i restanti al bordo inf.
    intervals_user = [ {[0, tf]}, allpar_tab(1:p) ];
    fixv = cellfun(@(x) x(1), allpar_tab(p+1:end));
    fixc = num2cell(fixv);
    Qh   = @(varargin) evalQ_perm(nreplicas, par_order, varargin{:}, fixc{:});

    core_dims = core_rank * ones(1, d);
    [ot, ol, per_lev] = observed_entries(core_dims, ncheb, coeff_levels, d);
    obs_last(id) = ol; obs_tot(id) = ot; tot_entr(id) = ncheb(end)^d;

    appendlog(logfile, sprintf('d=%d free: %s | fixed(lower bound): %s', d, ...
        strjoin(tabnames(1:p), ', '), fixstr(p, fixv)));
    appendlog(logfile, sprintf(['  observed entries: final level %.3e (%.2e%%) | ' ...
        'cumulative %.3e (%.2e%%) | total %d^%d = %.3e'], ...
        ol, 100*ol/tot_entr(id), ot, 100*ot/tot_entr(id), ncheb(end), d, tot_entr(id)));

    clear options_levels
    for jl = 1:nlevels
        options_levels(jl).maxiter           = maxiter;
        options_levels(jl).maxinner          = maxinner;
        options_levels(jl).tolgradnorm       = tolgradnorm;
        options_levels(jl).minstepsize       = 1e-15;
        options_levels(jl).check_derivatives = false;
        options_levels(jl).solver            = 'trustregions';
        options_levels(jl).hessian           = 'gn';
        options_levels(jl).verbosity         = 0;
    end

    rng(run_seed);
    t0 = tic;
    Xfinal = run_multilevel(core_dims, ncheb, intervals_user, pi0, r, ...
                            options_levels, coeff_levels, Qh);
    time_cr(id) = toc(t0);

    N = ncheb(end);
    grids = cell(1, d);
    for j = 1:d, grids{j} = chebpts(N, intervals_user{j}); end
    rng(sample_seed);
    subs = zeros(nsamples, d);
    for j = 1:d, subs(:, j) = randi(N, nsamples, 1); end
    vals_true = zeros(nsamples, 1);
    for k = 1:nsamples
        prm = cell(1, d-1);
        for j = 2:d, prm{j-1} = grids{j}(subs(k, j)); end
        v = fiber_time(grids{1}, prm, pi0, r, [], Qh);   % ESATTO (expm)
        vals_true(k) = v(subs(k, 1));
    end
    vals_cr = eval_ttensor(Xfinal, subs);
    err_L2(id)   = norm(vals_cr - vals_true) / norm(vals_true);
    err_Linf(id) = max(abs(vals_cr - vals_true));
    done_d(id)   = true;

    save(outfile, 'd_list','ncheb','coeff_levels','core_rank','nsamples', ...
         'time_cr','err_L2','err_Linf','obs_last','obs_tot','tot_entr', ...
         'run_seed','sample_seed','maxiter','maxinner','tolgradnorm', ...
         'nreplicas','tf','par_order','allpar_tab','done_d');

    appendlog(logfile, sprintf('  d=%d: tempo=%.1f s | err L2=%.3e | err Linf=%.3e', ...
        d, time_cr(id), err_L2(id), err_Linf(id)));
end

%% ---- riepilogo (formato di tab:cheb_dsweep) ------------------------------
appendlog(logfile, '======== tab:cheb_dsweep ========');
appendlog(logfile, sprintf('%3s | %10s | %11s | %11s | %8s | %10s', ...
    'd','Accuracy','Observed','Total','Time (s)','Observed %'));
for id = 1:nd
    appendlog(logfile, sprintf('%3d | %10.3e | %11.3e | %11.3e | %8.1f | %10.3e', ...
        d_list(id), err_L2(id), obs_last(id), tot_entr(id), time_cr(id), ...
        100*obs_last(id)/tot_entr(id)));
end
appendlog(logfile, 'DONE');
fprintf('\nSalvati risultati in %s\n', outfile);
end

%% ===== funzioni locali ====================================================
function v = getdef(cfg, f, default)
    if isfield(cfg, f), v = cfg.(f); else, v = default; end
end

function tf = isAbsPath(p)
    tf = ~isempty(regexp(p, '^([A-Za-z]:[\\/]|[\\/])', 'once'));
end

function appendlog(f,msg)
    fid=fopen(f,'a'); fprintf(fid,'%s  %s\n', datestr(now,'HH:MM:SS'), msg); fclose(fid);
    fprintf('%s\n', msg);
end

% Rimette i valori (in ordine di ATTIVAZIONE) nelle posizioni degli argomenti
% di evalQ_extended(nr, lambda, lambda2, mu, mu_d, cf, c2, c1, cr).
function Q = evalQ_perm(nr, par_order, varargin)
    vals = [varargin{:}];
    v = zeros(1, 8); v(par_order) = vals;
    vc = num2cell(v);
    Q = evalQ_extended(nr, vc{:});
end

function [tot, last, per_lev] = observed_entries(core_dims, ncheb, coeff_levels, d)
    per_lev = zeros(1, numel(ncheb));
    for lev = 1:numel(ncheb)
        n = ncheb(lev);
        if lev == 1
            per_lev(lev) = n^d;
        else
            nr      = round(coeff_levels(lev) * (sum(core_dims)*n + prod(core_dims)));
            nfibers = min(ceil(nr / n), n^(d-1));
            per_lev(lev) = nfibers * n;
        end
    end
    tot  = sum(per_lev);
    last = per_lev(end);
end

% nomi nell'ordine di tab:case1_parameters (par_2..par_9)
function nm = tabnames(k)
    all = {'lambda','cf','cr','mu_d','mu','lambda2','c1','c2'};
    nm = all(k);
end

function s = fixstr(p, fixv)
    all = {'lambda','cf','cr','mu_d','mu','lambda2','c1','c2'};
    if p >= 8, s = '(nessuno)'; return; end
    names = all(p+1:8);
    parts = arrayfun(@(i) sprintf('%s=%.3g', names{i}, fixv(i)), ...
        1:numel(names), 'UniformOutput', false);
    s = strjoin(parts, ', ');
end

function Xfinal = run_multilevel(core_dims, ncheb, intervals, pi0, en, ...
                                 options_levels, coeff_levels, Qh)
    d = numel(intervals);
    nlevels = numel(ncheb);
    Xprev = [];
    for lev = 1:nlevels
        n = ncheb(lev);
        values = cell(1, d);
        for i = 1:d, values{i} = chebpts(n, intervals{i}); end
        opts  = options_levels(lev);
        coeff = coeff_levels(lev);
        if isempty(Xprev)
            [~, X] = cheb_riemm_sparse(core_dims, n*ones(1,d), n, values, ...
                intervals, pi0, en, opts, coeff, true, [], [], Qh);
        else
            [~, X] = cheb_riemm_sparse(core_dims, n*ones(1,d), ncheb(max(lev-1,1)), ...
                values, intervals, pi0, en, opts, coeff, false, Xprev, [], Qh);
        end
        Xprev = X;
    end
    Xfinal = Xprev;
end

function vals = eval_ttensor(X, subs)
    ns = size(subs,1); dd = size(subs,2);
    vals = zeros(ns,1);
    for k = 1:ns
        cols = cell(1, dd);
        for m = 1:dd, cols{m} = X.U{m}(subs(k,m), :).'; end
        vals(k) = double(ttv(X.core, cols, 1:dd));
    end
end
