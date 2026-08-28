function Script_case2_compare_reliability_seed50(cfg)
%SCRIPT_CASE2_COMPARE_RELIABILITY_SEED50  Case study 2, reliability, 50 seeds.
%   Compares mixed-rank Chebyshev + Riemannian completion against ACA on
%   the IPS model for d = 5, 6, 7 at a given Erlang order NE, over 50 seeds.
%
%   Script_case2_compare_reliability_seed50(struct('ne',20))
%
%   Writes case2_compare_reliability_seed50_d567_ne<ne>.mat.

here = fileparts(mfilename('fullpath'));
% Le funzioni condivise stanno in <repo>/src, messe sul path da setup_paths.m
%  (gli script vivono in <repo>/experiments/<esperimento>/).
repoRoot = fileparts(fileparts(here));
run(fullfile(repoRoot, 'setup_paths.m'));
if ~exist('chebpts','file'),      error('Serve Chebfun sul path.'); end
if ~exist('tenrand','file'),      error('Serve Tensor Toolbox (>= 2.6) sul path.'); end
if ~exist('trustregions','file'), error('Serve Manopt sul path.'); end

if nargin < 1, cfg = struct(); end
g = @(f,v) getdef(cfg,f,v);

%% ---- CONFIG --------------------------------------------------------------
d_list       = g('d_list', [5 6 7]);
seed_list    = g('seed_list', 1:50);
ne           = g('ne', 20);                % fasi Erlang: 20, 30 o 40
tol          = g('tol', 1e-5);
Nt           = g('Nt', 64);
Np           = g('Np', 64);
nsamples     = g('nsamples', 2000);
sample_seed  = g('sample_seed', 0);        % test set FISSO
do_cheb      = g('do_cheb', true);

% Cheb (rango misto)
rank_time    = g('rank_time', 7);
rank_par     = g('rank_par', 3);
nt_levels    = g('nt_levels', [7 16 32 64]);
np_levels    = g('np_levels', [3 16 32 64]);
coeff_levels = g('coeff_levels', [1 2 10 15]);
maxiter      = g('maxiter', 100);
maxinner     = g('maxinner', 20);
tolgradnorm  = g('tolgradnorm', 1e-7);
solver       = g('solver', 'trustregions');

aca_maxit    = 1000;

outfile = g('outfile', sprintf('case2_compare_reliability_seed50_d567_ne%d.mat', ne));
logfile = g('logfile', sprintf('compare_reliability_seed50_ne%d_progress.txt', ne));
if ~isAbsPath(outfile), outfile = fullfile(here, outfile); end
if ~isAbsPath(logfile), logfile = fullfile(here, logfile); end
resume           = g('resume', false);
retime_outliers  = g('retime_outliers', false);
outlier_factor   = g('outlier_factor', 10);
make_figures     = g('make_figures', true);
figprefix        = g('figprefix', sprintf('case2_seed50_ne%d', ne));

% coerenza griglie multilivello con la griglia finale comune
if nt_levels(1) ~= rank_time, error('nt_levels(1) deve = rank_time'); end
if np_levels(1) ~= rank_par,  error('np_levels(1) deve = rank_par'); end
if nt_levels(end) ~= Nt, error('nt_levels(end) deve = Nt (%d)', Nt); end
if np_levels(end) ~= Np, error('np_levels(end) deve = Np (%d)', Np); end
if numel(coeff_levels) ~= numel(nt_levels) || numel(np_levels) ~= numel(nt_levels)
    error('nt_levels, np_levels and coeff_levels must have the same length');
end
% Dal livello 2 in poi la griglia e' [nt_levels(lev), np_levels(lev)*ones]:
% i ranghi del core devono starci dentro, altrimenti il solver esplode con un
% errore di dimensioni incomprensibile dentro Manopt.
if numel(nt_levels) > 1
    if rank_time > min(nt_levels(2:end))
        error('rank_time = %d > min(nt_levels(2:end)) = %d', rank_time, min(nt_levels(2:end)));
    end
    if rank_par > min(np_levels(2:end))
        error('rank_par = %d > min(np_levels(2:end)) = %d', rank_par, min(np_levels(2:end)));
    end
end

%% ---- Modello (IPS, Fig. 4), misura eq11 ----------------------------------
nstates = 3 + 2*ne;
tf  = 24*365*10;
pi0 = zeros(nstates,1); pi0(1)=1;
en  = zeros(nstates,1); en(nstates)=1;
fun = @measure_eq11;

allpar_eq10 = { [1, 3],       [1e-4, 1e-3], [1e-5, 1e-4], [1e-5, 1e-4], ...
                [1e-6, 1e-5], [0.5, 2.5],   [0.25, 0.75], [1e-6, 1e-5], ...
                [1e-7, 1e-5], [1e-6, 1e-5], [1e-6, 1e-5], [1e-7, 1e-6] };
par_order = g('par_order', [3 4 5 7 8 9 10 11 12 6 1 2]);
allpar    = allpar_eq10(par_order);

% scope condiviso con la nested Afiber
grids = {}; n = []; p = 0; d = 0; Qh = [];
aca_nft = 0;                     % contatore entrate osservate da ACA

nd = numel(d_list); nseed = numel(seed_list);
aca_rk  = nan(nseed, nd);  aca_err = nan(nseed, nd);  aca_t = nan(nseed, nd);
aca_cap = false(nseed, nd);
che_err = nan(nseed, nd);  che_t = nan(nseed, nd);
done    = false(nseed, nd);
% entrate osservate: per ACA variano col seme (rango adattivo), per Cheb sono
% deterministiche dalla configurazione, quindi basta un valore per d.
aca_obs = nan(nseed, nd);  che_obs = nan(1, nd);
time_outliers = false(nseed, nd);

%% ---- resume --------------------------------------------------------------
if resume && isfile(outfile)
    S = load(outfile);
    okcfg = isequal(S.d_list(:).', d_list(:).') && ...
            isequal(S.seed_list(:).', seed_list(:).') && S.ne == ne && ...
            S.Nt == Nt && S.Np == Np && S.nsamples == nsamples && ...
            S.rank_time == rank_time && S.rank_par == rank_par;
    if ~okcfg
        error(['resume=true but %s was produced with a different cfg. ' ...
               'Delete it or change outfile.'], outfile);
    end
    aca_rk=S.aca_rk; aca_err=S.aca_err; aca_t=S.aca_t; aca_cap=S.aca_cap;
    che_err=S.che_err; che_t=S.che_t; done=S.done;
    if isfield(S,'aca_obs'), aca_obs=S.aca_obs; end
    if isfield(S,'che_obs'), che_obs=S.che_obs; end
    fid = fopen(logfile,'a');
else
    fid = fopen(logfile,'w');
end
fprintf(fid, ['start %s | ne=%d d_list=[%s] nseeds=%d Nt=%d Np=%d ' ...
    'core=[%d %d..] nt=[%s] np=[%s] coeff=[%s] nsamples=%d\n'], ...
    datestr(now), ne, num2str(d_list), nseed, Nt, Np, rank_time, rank_par, ...
    num2str(nt_levels), num2str(np_levels), num2str(coeff_levels), nsamples);
fclose(fid);
if resume
    appendlog(logfile, sprintf('RESUME: %d/%d celle gia'' fatte.', nnz(done), numel(done)));
    if retime_outliers
        bad = flag_time_outliers(che_t, aca_t, done, outlier_factor);
        if any(bad(:))
            [ri,ci] = find(bad);
            appendlog(logfile, sprintf(['RETIME: %d cells with time above %gx the ' ...
                'median will be re-measured: %s'], nnz(bad), outlier_factor, ...
                strjoin(arrayfun(@(a,b)sprintf('(seed %d,d=%d)', seed_list(a), d_list(b)), ...
                ri, ci, 'UniformOutput', false), ' ')));
            done(bad) = false;
        else
            appendlog(logfile, 'RETIME: nessun outlier di tempo da rimisurare.');
        end
    end
end

t_run = tic;

for id = 1:nd
    d = d_list(id);  p = d - 1;
    intervals_user = [ {[0, tf]}, allpar(1:p) ];
    fixc = num2cell(cellfun(@(iv) iv(1), allpar(p+1:end)));
    Qh = @(varargin) evalQ_perm(ne, par_order, varargin{:}, fixc{:});

    grids = cell(1,d);
    grids{1} = chebpts(Nt, intervals_user{1});
    for j = 2:d, grids{j} = chebpts(Np, intervals_user{j}); end
    n = [Nt, Np*ones(1,p)];

    if all(done(:,id))
        appendlog(logfile, sprintf('===== d=%d (p=%d, ne=%d) gia'' completo, salto =====', d, p, ne));
        continue
    end

    % test set FISSO (una volta per d)
    t_ts = tic;
    rng(sample_seed);
    subs = zeros(nsamples, d); subs(:,1) = randi(Nt, nsamples, 1);
    for j = 2:d, subs(:,j) = randi(Np, nsamples, 1); end
    vals_true = zeros(nsamples,1);
    for k = 1:nsamples
        prm = cell(1,p); for j = 2:d, prm{j-1} = grids{j}(subs(k,j)); end
        v = fiber_time(grids{1}, prm, pi0, en, fun, Qh); vals_true(k) = v(subs(k,1));
    end
    ntrue = norm(vals_true);
    dofACA = @(rk) rk * sum(n);
    appendlog(logfile, sprintf(['===== d=%d (p=%d, ne=%d) test set pronto ' ...
        '(ntrue=%.3e, %.1f s) — %d semi ====='], d, p, ne, ntrue, toc(t_ts), nseed));

    % opzioni Cheb
    core_dims = [rank_time, rank_par*ones(1,p)];
    clear opt
    for jl = 1:numel(nt_levels)
        opt(jl).maxiter=maxiter; opt(jl).maxinner=maxinner; opt(jl).tolgradnorm=tolgradnorm;
        opt(jl).minstepsize=1e-15; opt(jl).check_derivatives=false; opt(jl).solver=solver;
        opt(jl).hessian='gn'; opt(jl).verbosity=0;
    end

    % entrate osservate da Cheb: deterministiche dalla configurazione, quindi
    % le calcoliamo qui invece di strumentare cheb_riemm_sparse_mixed, che e'
    % condiviso con molti altri script.
    che_obs(id) = cheb_observed_entries(core_dims, nt_levels, np_levels, coeff_levels, d);
    appendlog(logfile, sprintf(['  entries observed by Cheb (deterministic): ' ...
        '%.3e out of %.3e total (%.2e%%)'], che_obs(id), Nt*Np^p, 100*che_obs(id)/(Nt*Np^p)));

    for is = 1:nseed
        if done(is,id), continue; end
        s = seed_list(is);

        % ----- ACA con seme s -----
        rng(s);
        aca_nft = 0;
        t0 = tic; U = {};
        evalc('U = aca_nd(n, @Afiber, tol);');
        aca_t(is,id) = toc(t0);
        aca_obs(is,id) = aca_nft;
        rk = size(U{1},2);
        Vv = ones(nsamples, rk); for j = 1:d, Vv = Vv .* U{j}(subs(:,j),:); end
        aca_rk(is,id)  = rk;
        aca_err(is,id) = norm(sum(Vv,2) - vals_true) / ntrue;
        aca_cap(is,id) = (rk >= aca_maxit);

        % ----- Cheb con seme s -----
        if do_cheb
            rng(s);
            t0 = tic; Xf = [];
            evalc(['Xf = run_multilevel_mixed(core_dims, nt_levels, np_levels, ' ...
                   'intervals_user, pi0, en, opt, coeff_levels, fun, Qh);']);
            che_t(is,id) = toc(t0);
            che_err(is,id) = norm(eval_ttensor(Xf, subs) - vals_true) / ntrue;
        end

        done(is,id) = true;

        % ---- salvataggio INCREMENTALE (dopo ogni seme) --------------------
        save_partial();

        if do_cheb
            appendlog(logfile, sprintf(['  seed %d/%d (rng=%d): ACA rank=%d%s L2=%.3e (%.0fs) | ' ...
                'Cheb L2=%.3e (%.0fs) | %s'], is, nseed, s, rk, tern(aca_cap(is,id),'*',''), ...
                aca_err(is,id), aca_t(is,id), che_err(is,id), che_t(is,id), ...
                eta_str(done, che_t, aca_t, t_run)));
        else
            appendlog(logfile, sprintf('  seed %d/%d (rng=%d): ACA rank=%d%s L2=%.3e (%.0fs) | %s', ...
                is, nseed, s, rk, tern(aca_cap(is,id),'*',''), aca_err(is,id), ...
                aca_t(is,id), eta_str(done, che_t, aca_t, t_run)));
        end
    end

    rr = aca_rk(:,id);
    appendlog(logfile, sprintf(['  >>> ACA rank: median=%.0f [%d..%d] spread=%.1fx | ' ...
        'median dof=%d [%d..%d] | median L2=%.3e'], median(rr,'omitnan'), min(rr), max(rr), ...
        max(rr)/min(rr), dofACA(round(median(rr,'omitnan'))), dofACA(min(rr)), dofACA(max(rr)), ...
        median(aca_err(:,id),'omitnan')));
    if do_cheb
        appendlog(logfile, sprintf('  >>> Cheb L2: median=%.3e spread=%.1fx (fixed rank, dof=%d)', ...
            median(che_err(:,id),'omitnan'), ...
            max(che_err(:,id))/min(che_err(:,id)), ...
            prod(core_dims)+Nt*rank_time+p*Np*rank_par));
        appendlog(logfile, sprintf(['  >>> observed entries: Cheb %.3e (fixed) | ' ...
            'ACA median %.3e [%.3e..%.3e] -> median ACA/Cheb %.1fx'], ...
            che_obs(id), median(aca_obs(:,id),'omitnan'), min(aca_obs(:,id)), ...
            max(aca_obs(:,id)), median(aca_obs(:,id),'omitnan')/che_obs(id)));
    end
end

%% ---- outlier di tempo ----------------------------------------------------
time_outliers = flag_time_outliers(che_t, aca_t, done, outlier_factor);
if any(time_outliers(:))
    [ri,ci] = find(time_outliers);
    appendlog(logfile, sprintf(['WARNING: %d cells with time above %gx the column ' ...
        'median (possible standby or external load): %s'], nnz(time_outliers), ...
        outlier_factor, strjoin(arrayfun(@(a,b) ...
        sprintf('(seme %d, d=%d, Cheb %.0fs, ACA %.0fs)', seed_list(a), d_list(b), ...
        che_t(a,b), aca_t(a,b)), ri, ci, 'UniformOutput', false), ' ')));
    appendlog(logfile, ['  -> rimisurabili con: cfg.resume = true, ' ...
        'cfg.retime_outliers = true']);
else
    appendlog(logfile, 'Nessun outlier di tempo: misure pulite.');
end
save_partial();

%% ---- riepilogo -----------------------------------------------------------
appendlog(logfile, sprintf('========== RIEPILOGO ne=%d (%d semi) ==========', ne, nseed));
appendlog(logfile, sprintf('%3s | %10s %9s %6s | %10s %9s %8s %6s', ...
    'd','Cheb L2','Cheb dof','t med','ACA L2','ACA dof','rk range','t med'));
for id = 1:nd
    pl = d_list(id)-1;
    appendlog(logfile, sprintf('%3d | %10.3e %9d %6.0f | %10.3e %9d %3d-%-4d %6.0f', ...
        d_list(id), median(che_err(:,id),'omitnan'), ...
        prod([rank_time, rank_par*ones(1,pl)])+Nt*rank_time+pl*Np*rank_par, ...
        median(che_t(:,id),'omitnan'), median(aca_err(:,id),'omitnan'), ...
        round(median(aca_rk(:,id),'omitnan'))*(Nt+pl*Np), ...
        min(aca_rk(:,id)), max(aca_rk(:,id)), median(aca_t(:,id),'omitnan')));
end
appendlog(logfile, '(* = ACA hit the maxit=1000 cap -> NO convergence)');
appendlog(logfile, sprintf('tempo totale: %.0f s (%.2f h)', toc(t_run), toc(t_run)/3600));
appendlog(logfile, 'DONE');

%% ---- figure diagnostiche -------------------------------------------------
if make_figures && do_cheb
    col_cr  = [0 0.45 0.74];
    col_aca = [0.85 0.33 0.10];

    f1 = figure('Visible','off','Position',[100 100 820 500]); hold on
    for id = 1:nd
        draw_box(id-0.18, che_t(:,id), col_cr);
        draw_box(id+0.18, aca_t(:,id), col_aca);
    end
    set(gca,'YScale','log'); grid on
    xlim([0.4 nd+0.6]); xticks(1:nd); xticklabels(arrayfun(@(x)sprintf('d=%d',x), d_list, 'UniformOutput',false));
    ylabel('computing time (s)');
    title(sprintf('Case study 2 (IPS), ne = %d : times over %d seeds', ne, nseed));
    saveas(f1, fullfile(here, [figprefix '_time_boxes.png']));

    f2 = figure('Visible','off','Position',[100 100 820 500]); hold on
    for id = 1:nd
        pl = d_list(id)-1;
        dof_c = prod([rank_time, rank_par*ones(1,pl)]) + Nt*rank_time + pl*Np*rank_par;
        draw_box(id-0.18, dof_c*ones(nnz(done(:,id)),1), col_cr);
        draw_box(id+0.18, aca_rk(:,id)*(Nt+pl*Np),       col_aca);
    end
    set(gca,'YScale','log'); grid on
    xlim([0.4 nd+0.6]); xticks(1:nd); xticklabels(arrayfun(@(x)sprintf('d=%d',x), d_list, 'UniformOutput',false));
    ylabel('degrees of freedom (DoF)');
    title(sprintf('Case study 2 (IPS), ne = %d : DoF over %d seeds', ne, nseed));
    saveas(f2, fullfile(here, [figprefix '_dof_boxes.png']));
    close([f1 f2]);
end

fprintf('\nSalvati risultati in %s\n', outfile);

%% ===== nested ============================================================
    function save_partial()
        save(outfile, 'd_list','seed_list','ne','tol','Nt','Np','nsamples', ...
            'sample_seed','par_order','rank_time','rank_par','nt_levels', ...
            'np_levels','coeff_levels','done','time_outliers', ...
            'aca_rk','aca_err','aca_t','aca_cap','che_err','che_t', ...
            'aca_obs','che_obs');
    end

    function v = Afiber(jm, i)
        prm = cell(1, p);
        for jj = 2:d, prm{jj-1} = grids{jj}(i(jj)); end
        if jm == 1
            v = fiber_time(grids{1}, prm, pi0, en, fun, Qh);
        else
            t = grids{1}(i(1)); v = zeros(n(jm),1);
            for q = 1:n(jm)
                prm{jm-1} = grids{jm}(q);
                vv = fiber_time(t, prm, pi0, en, fun, Qh); v(q) = vv(1);
            end
        end
        aca_nft = aca_nft + n(jm);      % entrate del tensore effettivamente valutate
        v = v(:);
    end
end

%% ===== locali =============================================================
function v = getdef(cfg,f,default), if isfield(cfg,f), v=cfg.(f); else, v=default; end, end
function s = tern(c,a,b), if c, s=a; else, s=b; end, end

function tf = isAbsPath(p)
    tf = ~isempty(regexp(p, '^([A-Za-z]:[\\/]|[\\/])', 'once'));
end

function appendlog(f,msg)
    fid=fopen(f,'a'); fprintf(fid,'%s  %s\n', datestr(now,'HH:MM:SS'), msg); fclose(fid);
    fprintf('%s\n', msg);
end

% Entrate del tensore osservate dal completamento multilivello. Replica la
% logica di cheb_riemm_sparse_mixed (righe ~52-70): al livello base si prende
% l'intera griglia a forma di core, ai livelli successivi si scelgono
% nfibers = ceil(nr / n1) fibre con nr = coeff*(sum(core.*dims) + prod(core)),
% saturato al numero massimo di fibre disponibili.
function tot = cheb_observed_entries(core_dims, nt_levels, np_levels, coeff_levels, d)
    tot = prod(core_dims);                       % livello 1: base_level = true
    for lev = 2:numel(nt_levels)
        dims      = [nt_levels(lev), np_levels(lev)*ones(1,d-1)];
        maxfibers = prod(dims(2:end));
        nr        = round(coeff_levels(lev) * (sum(core_dims .* dims) + prod(core_dims)));
        nfibers   = min(ceil(nr / dims(1)), maxfibers);
        tot       = tot + nfibers * dims(1);
    end
end

% Celle il cui tempo (Cheb o ACA) supera factor volte la mediana della propria
% colonna: tipicamente standby del sistema o carico esterno, non calcolo vero.
function bad = flag_time_outliers(che_t, aca_t, done, factor)
    bad = false(size(done));
    for c = 1:size(done,2)
        v = done(:,c);
        if nnz(v) < 5, continue; end        % mediana non affidabile
        mc = median(che_t(v,c),'omitnan');
        ma = median(aca_t(v,c),'omitnan');
        if ~isnan(mc), bad(:,c) = bad(:,c) | (v & che_t(:,c) > factor*mc); end
        if ~isnan(ma), bad(:,c) = bad(:,c) | (v & aca_t(:,c) > factor*ma); end
    end
end

function s = eta_str(done, che_t, aca_t, t_run)
    per = mean(che_t + aca_t, 1, 'omitnan');       % s/seme per ogni d
    lastk = find(~isnan(per), 1, 'last');
    if isempty(lastk), s = ''; return; end
    per(isnan(per)) = per(lastk);
    remaining = sum(per .* sum(~done, 1));
    s = sprintf('elapsed %.0f min, ETA %.0f min', toc(t_run)/60, remaining/60);
end

function Q = evalQ_perm(ne, par_order, varargin)
    v = zeros(1, numel(par_order)); v(par_order) = [varargin{:}];
    vc = num2cell(v); Q = evalQ_ips(ne, vc{:});
end

function v = measure_eq11(tvec, Q, pi0, r)
    Qt = full(Q)'; pi0 = pi0(:); r = r(:); tv = reshape(tvec,1,[]);
    [V,D] = eig(Qt,'vector'); ok = cond(V) < 1e8;
    if ok
        w = V\pi0; z = (r'*V).'; v = real(exp(D*tv).' * (z.*w));
        [tmx,imx] = max(tv);
        if tmx>0, ok = abs(v(imx) - r'*(expm(Qt*tmx)*pi0)) <= 1e-8; end
    end
    if ~ok, v = zeros(numel(tv),1); for k=1:numel(tv), v(k)=r'*(expm(Qt*tv(k))*pi0); end, end
end

function Xfinal = run_multilevel_mixed(core_dims, nt_levels, np_levels, ...
                        intervals, pi0, en, options_levels, coeff_levels, fun, Qh)
    d = numel(intervals); nlevels = numel(nt_levels); Xprev = [];
    for lev = 1:nlevels
        nt = nt_levels(lev); np = np_levels(lev);
        if lev==1, tensor_dims = core_dims; else, tensor_dims = [nt, np*ones(1,d-1)]; end
        values = cell(1,d); for i=1:d, values{i} = chebpts(tensor_dims(i), intervals{i}); end
        if isempty(Xprev)
            [~,X] = cheb_riemm_sparse_mixed(core_dims, tensor_dims, np, values, ...
                intervals, pi0, en, options_levels(lev), coeff_levels(lev), true, [], fun, Qh);
        else
            [~,X] = cheb_riemm_sparse_mixed(core_dims, tensor_dims, np, values, ...
                intervals, pi0, en, options_levels(lev), coeff_levels(lev), false, Xprev, fun, Qh);
        end
        Xprev = X;
    end
    Xfinal = Xprev;
end

function vals = eval_ttensor(X, subs)
    ns = size(subs,1); dd = size(subs,2); vals = zeros(ns,1);
    for k = 1:ns
        cols = cell(1,dd); for m=1:dd, cols{m} = X.U{m}(subs(k,m),:).'; end
        vals(k) = double(ttv(X.core, cols, 1:dd));
    end
end

function draw_box(x, data, col)
    data = data(~isnan(data));
    if isempty(data), return; end
    q = quantile(data, [0.25 0.5 0.75]);
    jit = (rand(numel(data),1)-0.5)*0.14;
    scatter(x+jit, data, 18, col, 'filled', 'MarkerFaceAlpha',0.45);
    plot([x-0.16 x+0.16], [q(2) q(2)], '-', 'Color',col, 'LineWidth',2.5);
    plot([x-0.12 x+0.12 x+0.12 x-0.12 x-0.12], ...
         [q(1) q(1) q(3) q(3) q(1)], '-', 'Color',col, 'LineWidth',1.2);
    plot([x x], [min(data) max(data)], '-', 'Color',col, 'LineWidth',0.8);
end
