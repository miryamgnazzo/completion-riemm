function Script_case1_robustness_seed50(cfg)
%SCRIPT_CASE1_ROBUSTNESS_SEED50  Case study 1, reliability, 50 seeds.
%   Compares Chebyshev + Riemannian completion against ACA for d = 3, 4, 5,
%   repeating every configuration over 50 seeds. Both methods share the
%   same seed and the same fixed test set.
%
%   Script_case1_robustness_seed50
%   Script_case1_robustness_seed50(struct('resume',true))
%
%   Writes case1_robustness_seed50_results.mat.
if nargin < 1, cfg = struct(); end
g = @(f,v) getdef(cfg,f,v);

nseeds       = g('nseeds', 50);               % semi condivisi dai due metodi
p_list       = g('p_list', [2 3 4]);          % # parametri liberi -> d = p+1
aca_tol      = g('aca_tol', 1e-6);
nsamples     = g('nsamples', 1000);

core_rank    = g('core_rank', 3);
ncheb        = g('ncheb', [3 8 16 32]);
coeff_levels = g('coeff_levels', [1 2 5 10]);

maxiter      = g('maxiter', 50);
maxinner     = g('maxinner', 20);
tolgradnorm  = g('tolgradnorm', 1e-8);

sample_seed  = g('sample_seed', 0);           % seme del test set (FISSO)

scriptDir = fileparts(mfilename('fullpath'));

outfile      = g('outfile', 'case1_robustness_seed50_results.mat');
logfile      = g('logfile', 'case1_robustness_seed50_progress.txt');
if ~isAbsPath(outfile), outfile = fullfile(scriptDir, outfile); end
if ~isAbsPath(logfile), logfile = fullfile(scriptDir, logfile); end
resume       = g('resume', false);
make_figures = g('make_figures', true);
figprefix    = g('figprefix', 'case1_robustness_seed50');

if numel(coeff_levels) ~= numel(ncheb)
    error('coeff_levels must have the same length as ncheb');
end

N = ncheb(end);

%% ---- Path ----------------------------------------------------------------
% Le funzioni condivise stanno in <repo>/src, messe sul path da setup_paths.m
%  (gli script vivono in <repo>/experiments/<esperimento>/).
repoRoot = fileparts(fileparts(scriptDir));
run(fullfile(repoRoot, 'setup_paths.m'));
if ~exist('chebpts','file'),      error('Serve Chebfun sul path.'); end
if ~exist('tenrand','file'),      error('Serve Tensor Toolbox sul path.'); end
if ~exist('trustregions','file'), error('Serve Manopt sul path.'); end

%% ---- Modello CONDIVISO (Case study 1c, nr = 2) ---------------------------
nreplicas = 2;
nstates   = nreplicas + 2;
tf = 24*365*10;
pi0 = zeros(nstates,1); pi0(1) = 1;
r   = [ones(nreplicas+1,1); 0];

param_names     = {'lambda','cf','cr','mu_d','mu'};
param_intervals = {[1e-6,1e-5],[0.9,0.99],[0.9,0.99],[0.25,0.75],[0.25,0.75]};

np = numel(p_list);

% risultati: righe = seme, colonne = p
cr_L2   = nan(nseeds, np);   cr_Linf = nan(nseeds, np);
cr_rank = nan(nseeds, np);   cr_time = nan(nseeds, np);
aca_L2   = nan(nseeds, np);  aca_Linf = nan(nseeds, np);
aca_rank = nan(nseeds, np);  aca_time = nan(nseeds, np);
aca_calls = nan(nseeds, np);
done = false(nseeds, np);

% ---- resume ---------------------------------------------------------------
if resume && isfile(outfile)
    S = load(outfile);
    okcfg = isequal(S.p_list(:).', p_list(:).') && S.nseeds == nseeds && ...
            isequal(S.ncheb(:).', ncheb(:).') && ...
            isequal(S.coeff_levels(:).', coeff_levels(:).') && ...
            S.core_rank == core_rank && S.nsamples == nsamples;
    if ~okcfg
        error(['resume=true but %s was produced with a different cfg. ' ...
               'Delete it or change outfile.'], outfile);
    end
    cr_L2=S.cr_L2; cr_Linf=S.cr_Linf; cr_rank=S.cr_rank; cr_time=S.cr_time;
    aca_L2=S.aca_L2; aca_Linf=S.aca_Linf; aca_rank=S.aca_rank;
    aca_time=S.aca_time; aca_calls=S.aca_calls; done=S.done;
    fid = fopen(logfile,'a');
else
    fid = fopen(logfile,'w');
end
fprintf(fid, 'start %s | nseeds=%d p_list=[%s] ncheb=[%s] coeff=[%s] nsamples=%d\n', ...
    datestr(now), nseeds, num2str(p_list), num2str(ncheb), ...
    num2str(coeff_levels), nsamples);
fclose(fid);
if resume
    appendlog(logfile, sprintf('RESUME: %d/%d celle gia'' fatte.', ...
        nnz(done), numel(done)));
end

grids = {};   nft = 0;   % condivise con la funzione annidata di conteggio
t_run = tic;

for ip = 1:np
    p = p_list(ip);
    d = p + 1;
    intervals_user = [ {[0, tf]}, param_intervals(1:p) ];

    appendlog(logfile, sprintf(['########## p = %d free parameters (d = %d) ' ...
        '##########  free: %s'], p, d, strjoin(param_names(1:p), ', ')));

    grids = cell(1, d);
    for j = 1:d, grids{j} = chebpts(N, intervals_user{j}); end

    % ---- TEST SET condiviso e FISSO --------------------------------------
    t_ts = tic;
    rng(sample_seed);
    subs = zeros(nsamples, d);
    for j = 1:d, subs(:, j) = randi(N, nsamples, 1); end
    vals_true = zeros(nsamples, 1);
    for k = 1:nsamples
        prm = cell(1, d-1);
        for j = 2:d, prm{j-1} = grids{j}(subs(k, j)); end
        v = fiber_time(grids{1}, prm, pi0, r);
        vals_true(k) = v(subs(k, 1));
    end
    nrm_true = norm(vals_true);
    appendlog(logfile, sprintf('  test set FISSO pronto (%d campioni, %.1f s).', ...
        nsamples, toc(t_ts)));

    core_dims = core_rank * ones(1, d);
    clear options_levels
    for jl = 1:numel(ncheb)
        options_levels(jl).maxiter           = maxiter;
        options_levels(jl).maxinner          = maxinner;
        options_levels(jl).tolgradnorm       = tolgradnorm;
        options_levels(jl).minstepsize       = 1e-15;
        options_levels(jl).check_derivatives = false;
        options_levels(jl).solver            = 'trustregions';
       % options_levels(jl).hessian           = 'gn';
        options_levels(jl).verbosity         = 0;
    end

    for s = 1:nseeds
        if done(s, ip), continue; end

        % ---- (A) Cheb + Riemann con seme s -------------------------------
        rng(s);
        Xuser = [];
        t0 = tic;
        evalc(['[~, Xuser] = cheb_completion_multi(core_dims, ncheb, ' ...
               'intervals_user, pi0, r, options_levels, coeff_levels);']);
        cr_time(s, ip) = toc(t0);
        vals_cr = eval_ttensor(Xuser, subs);
        cr_L2(s, ip)   = norm(vals_cr - vals_true) / nrm_true;
        cr_Linf(s, ip) = max(abs(vals_cr - vals_true));
        cr_rank(s, ip) = core_rank;             % fisso per costruzione

        % ---- (B) ACA con seme s ------------------------------------------
        rng(s);
        nft = 0;
        U = {};
        t0 = tic;
        evalc('U = aca_nd(N*ones(1,d), @Afiber_counted, aca_tol);');
        aca_time(s, ip) = toc(t0);
        vals_aca = eval_cp(U, subs);
        aca_L2(s, ip)    = norm(vals_aca - vals_true) / nrm_true;
        aca_Linf(s, ip)  = max(abs(vals_aca - vals_true));
        aca_rank(s, ip)  = size(U{1}, 2);
        aca_calls(s, ip) = nft;

        done(s, ip) = true;

        % ---- salvataggio INCREMENTALE (dopo ogni seme) -------------------
        save(outfile, 'p_list','nseeds','aca_tol','core_rank','ncheb', ...
             'coeff_levels','nsamples','sample_seed','done', ...
             'cr_L2','cr_Linf','cr_rank','cr_time', ...
             'aca_L2','aca_Linf','aca_rank','aca_calls','aca_time');

        appendlog(logfile, sprintf(['  p=%d seed %d/%d: Cheb L2=%.3e (%.0f s) | ' ...
            'ACA rank=%d L2=%.3e (%.0f s) | %s'], p, s, nseeds, ...
            cr_L2(s,ip), cr_time(s,ip), aca_rank(s,ip), aca_L2(s,ip), ...
            aca_time(s,ip), eta_str(done, cr_time, aca_time, t_run)));
    end

    appendlog(logfile, sprintf(['  >>> p=%d : Cheb L2 [med=%.2e spread=%.1fx] rank=%d | ' ...
        'ACA L2 [med=%.2e spread=%.1fx] rank=[%d..%d]'], p, ...
        median(cr_L2(:,ip),'omitnan'),  max(cr_L2(:,ip))/min(cr_L2(:,ip)), core_rank, ...
        median(aca_L2(:,ip),'omitnan'), max(aca_L2(:,ip))/min(aca_L2(:,ip)), ...
        min(aca_rank(:,ip)),  max(aca_rank(:,ip))));
end

%% ---- Tabella riassuntiva (spread = dispersione sui semi) -----------------
appendlog(logfile, sprintf('============ ROBUSTEZZA SIMMETRICA (%d semi) ============', nseeds));
appendlog(logfile, sprintf('%3s | %3s | %10s %10s %5s | %10s %10s %7s', ...
        'p','d','L2 med','spread','rank','L2 med','spread','rk max'));
for ip = 1:np
    appendlog(logfile, sprintf('%3d | %3d | %10.2e %9.1fx %5d | %10.2e %9.1fx %7d', ...
        p_list(ip), p_list(ip)+1, ...
        median(cr_L2(:,ip),'omitnan'),  max(cr_L2(:,ip))/min(cr_L2(:,ip)),  core_rank, ...
        median(aca_L2(:,ip),'omitnan'), max(aca_L2(:,ip))/min(aca_L2(:,ip)), ...
        max(aca_rank(:,ip))));
end
appendlog(logfile, sprintf('tempo totale: %.0f s (%.2f h)', toc(t_run), toc(t_run)/3600));
appendlog(logfile, 'DONE');

%% ---- Figure --------------------------------------------------------------
if make_figures
    xp = p_list;
    col_cr  = [0 0.45 0.74];      % blu  = Cheb+Riemann
    col_aca = [0.85 0.33 0.10];   % arancio = ACA

    % (1) errore L2 vs p: DUE nuvole (una per metodo) con mediana
    f1 = figure('Visible','off','Position',[100 100 760 500]);
    hold on
    for ip = 1:np
        scatter(repmat(xp(ip),nseeds,1)-0.06, cr_L2(:,ip), 18, col_cr, 'filled', ...
            'MarkerFaceAlpha',0.4, 'HandleVisibility','off');
        scatter(repmat(xp(ip),nseeds,1)+0.06, aca_L2(:,ip), 18, col_aca, 'filled', ...
            'MarkerFaceAlpha',0.4, 'HandleVisibility','off');
    end
    h_cr  = plot(xp, median(cr_L2,1,'omitnan'),  '-s', 'Color',col_cr,  'LineWidth',1.8, ...
            'MarkerFaceColor',col_cr);
    h_aca = plot(xp, median(aca_L2,1,'omitnan'), '-o', 'Color',col_aca, 'LineWidth',1.8, ...
            'MarkerFaceColor',col_aca);
    set(gca,'YScale','log'); grid on
    xlabel('number of free parameters  p'); ylabel('relative L2 error (test set)');
    title(sprintf('Error distribution over %d seeds (both methods)', nseeds));
    legend([h_cr h_aca], {'Cheb+Riemann','ACA'}, 'Location','east');
    xticks(xp);
    saveas(f1, fullfile(scriptDir, [figprefix '_error_vs_p.png']));

    % (2) box affiancati a p = max: le due distribuzioni fianco a fianco
    ipmax = np;
    f2 = figure('Visible','off','Position',[100 100 620 500]);
    hold on
    draw_box(1, cr_L2(:,ipmax),  col_cr);
    draw_box(2, aca_L2(:,ipmax), col_aca);
    set(gca,'YScale','log'); grid on
    xlim([0.4 2.6]); xticks([1 2]);
    xticklabels({'Cheb+Riemann','ACA'});
    ylabel('relative L2 error (test set)');
    title(sprintf('p = %d : side-by-side distributions (%d seeds)', p_list(ipmax), nseeds));
    saveas(f2, fullfile(scriptDir, [figprefix '_boxes.png']));

    % (3) rango vs p: Cheb fisso (per costruzione) vs nuvola ACA
    f3 = figure('Visible','off','Position',[100 100 760 500]);
    hold on
    for ip = 1:np
        scatter(repmat(xp(ip),nseeds,1)+0.06, aca_rank(:,ip), 18, col_aca, 'filled', ...
            'MarkerFaceAlpha',0.4, 'HandleVisibility','off');
    end
    h_aca = plot(xp, median(aca_rank,1,'omitnan'), '-o', 'Color',col_aca, 'LineWidth',1.6, ...
            'MarkerFaceColor',col_aca);
    h_cr  = plot(xp, core_rank*ones(1,np), '-s', 'Color',col_cr, 'LineWidth',1.8, ...
            'MarkerFaceColor',col_cr);
    grid on
    xlabel('number of free parameters  p'); ylabel('rank of the representation');
    title('Rank: Cheb+Riemann fixed by construction vs ACA discovered');
    legend([h_cr h_aca], {'Cheb+Riemann (fixed)','ACA (median over seeds)'}, ...
            'Location','northwest');
    xticks(xp);
    saveas(f3, fullfile(scriptDir, [figprefix '_rank_vs_p.png']));
    close([f1 f2 f3]);
end

fprintf('\nSalvati risultati in %s\n', outfile);

%% ===== funzione annidata (condivide grids/nft/pi0/r) =====================
    function f = Afiber_counted(j, i)
        dloc = numel(grids);
        prm = cell(1, dloc-1);
        for m = 2:dloc, prm{m-1} = grids{m}(i(m)); end
        if j == 1
            f = fiber_time(grids{1}, prm, pi0, r); f = f(:);
            nft = nft + 1;
        else
            Nj = numel(grids{j});
            f = zeros(Nj, 1);
            for qq = 1:Nj
                prm{j-1} = grids{j}(qq);
                v = fiber_time(grids{1}, prm, pi0, r);
                f(qq) = v(i(1));
            end
            nft = nft + Nj;
        end
    end
end

%% ===== funzioni locali ====================================================
function v = getdef(cfg, f, default)
    if isfield(cfg, f), v = cfg.(f); else, v = default; end
end

function tf = isAbsPath(p)
    tf = ~isempty(regexp(p, '^([A-Za-z]:[\\/]|[\\/])', 'once'));
end

function appendlog(f, msg)
    fid = fopen(f,'a'); fprintf(fid, '%s  %s\n', datestr(now,'HH:MM:SS'), msg); fclose(fid);
    fprintf('%s\n', msg);
end

% Stima del tempo residuo: usa il costo medio per seme gia' osservato in
% ciascuna colonna p; per le colonne non ancora iniziate usa l'ultima nota.
function s = eta_str(done, cr_time, aca_time, t_run)
    per = mean(cr_time + aca_time, 1, 'omitnan');    % s/seme per ogni p
    lastk = find(~isnan(per), 1, 'last');
    if isempty(lastk), s = ''; return; end
    per(isnan(per)) = per(lastk);
    remaining = sum(per .* sum(~done, 1));
    s = sprintf('elapsed %.0f min, ETA %.0f min', toc(t_run)/60, remaining/60);
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

function vals = eval_cp(U, subs)
    ns = size(subs,1); dd = size(subs,2); kk = size(U{1},2);
    vals = zeros(ns,1);
    for k = 1:ns
        w = ones(1, kk);
        for m = 1:dd, w = w .* U{m}(subs(k,m), :); end
        vals(k) = sum(w);
    end
end

function draw_box(x, data, col)
    % box "fatto a mano" (quartili + mediana) piu' nuvola dei punti, senza
    % dipendere dallo Statistics Toolbox.
    data = data(~isnan(data));
    q = quantile(data, [0.25 0.5 0.75]);
    n = numel(data);
    jit = (rand(n,1)-0.5)*0.18;
    scatter(x+jit, data, 26, col, 'filled', 'MarkerFaceAlpha',0.6);
    plot([x-0.3 x+0.3], [q(2) q(2)], '-', 'Color',col, 'LineWidth',2.5);
    plot([x-0.22 x+0.22 x+0.22 x-0.22 x-0.22], ...
         [q(1) q(1) q(3) q(3) q(1)], '-', 'Color',col, 'LineWidth',1.2);
    plot([x x], [min(data) max(data)], '-', 'Color',col, 'LineWidth',0.8, ...
         'HandleVisibility','off');
end
