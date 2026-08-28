function Script_case1_underrepair_d6ext_r3_seed50(cfg)
%SCRIPT_CASE1_UNDERREPAIR_D6EXT_R3_SEED50  Case study 1 under repair, d = 6.
%   Mean time under repair on the extended Case study 1 model, comparing
%   fixed-rank Chebyshev + Riemannian completion against ACA over 50 seeds.
%
%   Script_case1_underrepair_d6ext_r3_seed50
%   Script_case1_underrepair_d6ext_r3_seed50(struct('resume',true))
%
%   Writes d6ext_physt_r3_seed50_results.mat.
if nargin < 1, cfg = struct(); end
g = @(f,v) getdef(cfg,f,v);

core_dims    = g('core_dims', [16 3 3 3 3 3]);
ncheb        = g('ncheb', [16 16]);
coeff_levels = g('coeff_levels', [1 6]);
d            = g('d', 6);
nsamples     = g('nsamples', 1000);
aca_tol      = g('aca_tol', 1e-6);
sample_seed  = g('sample_seed', 0);              % test set FISSO
seed_list    = g('seed_list', 1:50);

maxiter      = g('maxiter', 120);
maxinner     = g('maxinner', 50);
tolgradnorm  = g('tolgradnorm', 1e-11);
storedepth   = g('storedepth', 3);

scriptDir = fileparts(mfilename('fullpath'));

outfile      = g('outfile', 'd6ext_physt_r3_seed50_results.mat');
logfile      = g('logfile', 'd6ext_physt_r3_seed50_progress.txt');
if ~isAbsPath(outfile), outfile = fullfile(scriptDir, outfile); end
if ~isAbsPath(logfile), logfile = fullfile(scriptDir, logfile); end
resume       = g('resume', false);
make_figures = g('make_figures', true);
figprefix    = g('figprefix', 'd6ext_physt_r3_seed50');

if numel(coeff_levels) ~= numel(ncheb)
    error('coeff_levels must have the same length as ncheb');
end
if numel(core_dims) ~= d
    error('core_dims deve avere %d elementi (d = %d)', d, d);
end
% Dal livello 2 in poi run_ml costruisce la griglia td = ncheb(lev)*ones(1,d):
% ogni rango del core deve starci dentro, altrimenti il fattore di modo e'
% rettangolare "al contrario" e il solver esplode con un errore di dimensioni
% incomprensibile dentro Manopt. Meglio fermarsi subito con un messaggio utile.
if numel(ncheb) > 1 && any(core_dims > min(ncheb(2:end)))
    error(['core_dims = [%s] is not compatible with ncheb = [%s]: from ' ...
           'level 2 on the grid has min(ncheb(2:end)) = %d points per ' ...
           'direction, so every rank must be <= %d.'], ...
           num2str(core_dims), num2str(ncheb), min(ncheb(2:end)), min(ncheb(2:end)));
end

%% ---- Path ----------------------------------------------------------------
% Le funzioni condivise stanno in <repo>/src, messe sul path da setup_paths.m
%  (gli script vivono in <repo>/experiments/<esperimento>/).
repoRoot = fileparts(fileparts(scriptDir));
run(fullfile(repoRoot, 'setup_paths.m'));
if ~exist('chebpts','file'),      error('Serve Chebfun sul path.'); end
if ~exist('tenrand','file'),      error('Serve Tensor Toolbox sul path.'); end
if ~exist('trustregions','file'), error('Serve Manopt sul path.'); end

%% ---- Modello (Case study 1 ESTESO, nr = 3), misura Repair ----------------
nreplicas = 3; nstates = nreplicas + 2;   % nr=3 (estensione attiva)
tf = 24*365*10;
pi0 = zeros(nstates,1); pi0(1)=1;
en  = zeros(nstates,1); en(2:nreplicas)=1;

lam2_fix = 1e-7; c1_fix = 0.95; c2_fix = 0.80;
Qh = @(la,cf,cr,mud,mu) evalQ_extended(nreplicas, la, lam2_fix, mu, mud, cf, c2_fix, c1_fix, cr);
param_iv = { [1e-6,1e-5], [0.90,0.99], [0.90,0.99], [0.25,0.75], [0.25,0.75] };

tmap = @(s) s;                            % tempo FISICO (nessun log-mapping)
N = ncheb(end);
p = d-1;
if numel(param_iv) < p
    error('at least %d parameter intervals are required for d = %d', p, d);
end
intervals = [ {[0, tf]}, param_iv(1:p) ];

% normalizzazione della misura al punto medio dello spazio dei parametri
midc = num2cell(cellfun(@(iv)0.5*(iv(1)+iv(2)), param_iv(1:p)));
Q0 = Qh(midc{:});
s0 = max(abs(ur(tmap(chebpts(N,[0,tf])), Q0, pi0, en)));
fun = @(s,Q,p0,e) ur(tmap(s),Q,p0,e)/s0;

%% ---- risultati + resume --------------------------------------------------
ns_seed = numel(seed_list);
err_cheb = nan(ns_seed,1);  tim_cheb = nan(ns_seed,1);
err_aca  = nan(ns_seed,1);  tim_aca  = nan(ns_seed,1);  k_aca = nan(ns_seed,1);
done     = false(ns_seed,1);

if resume && isfile(outfile)
    S = load(outfile);
    okcfg = isequal(S.seed_list(:).', seed_list(:).') && ...
            isequal(S.core_dims(:).', core_dims(:).') && ...
            isequal(S.ncheb(:).', ncheb(:).') && ...
            isequal(S.coeff_levels(:).', coeff_levels(:).') && ...
            S.d == d && S.nsamples == nsamples && S.nreplicas == nreplicas;
    if ~okcfg
        error(['resume=true but %s was produced with a different cfg. ' ...
               'Delete it or change outfile.'], outfile);
    end
    err_cheb=S.err_cheb; tim_cheb=S.tim_cheb; err_aca=S.err_aca;
    tim_aca=S.tim_aca;   k_aca=S.k_aca;       done=S.done;
    fid = fopen(logfile,'a');
else
    fid = fopen(logfile,'w');
end
fprintf(fid, ['start %s | d=%d nr=%d core=[%s] ncheb=[%s] coeff=[%s] ' ...
    'nsamples=%d nseeds=%d\n'], datestr(now), d, nreplicas, num2str(core_dims), ...
    num2str(ncheb), num2str(coeff_levels), nsamples, ns_seed);
fclose(fid);
if resume
    appendlog(logfile, sprintf('RESUME: %d/%d semi gia'' fatti.', nnz(done), ns_seed));
end

%% ---- test set FISSO (una volta sola) -------------------------------------
t_ts = tic;
% NB: la cella si chiama 'grd', non 'grid', per non oscurare la funzione
% grid() usata nel blocco delle figure piu' sotto.
grd = cell(1,d); for i=1:d, grd{i}=chebpts(N,intervals{i}); end
rng(sample_seed);
subs = zeros(nsamples,d); for j=1:d, subs(:,j)=randi(N,nsamples,1); end
vt = zeros(nsamples,1);
for k=1:nsamples
    prm=cell(1,d-1); for j=2:d, prm{j-1}=grd{j}(subs(k,j)); end
    v=fiber_time(grd{1},prm,pi0,en,fun,Qh); vt(k)=v(subs(k,1));
end
nvt = norm(vt);
Af = @(j,i) fib_od(j,i,grd,pi0,en,fun,Qh);
appendlog(logfile, sprintf(['d=%d ESTESO r3 nr=%d  test set FISSO pronto ' ...
    '(nvt=%.3e, %.1f s).'], d, nreplicas, nvt, toc(t_ts)));

%% ---- opzioni ottimizzazione (fisse) --------------------------------------
clear opt
for j=1:numel(ncheb)
    opt(j).maxiter=maxiter; opt(j).maxinner=maxinner; opt(j).tolgradnorm=tolgradnorm;
    opt(j).minstepsize=1e-15; opt(j).check_derivatives=false;
    opt(j).solver='trustregions'; opt(j).hessian='gn'; opt(j).storedepth=storedepth;
end

%% ---- loop sui semi (ORDINE: prima ACA, poi Cheb) -------------------------
t_run = tic;
for is = 1:ns_seed
    if done(is), continue; end
    s = seed_list(is);

    % (A) ACA con seme s (pivot casuali al check di convergenza)
    rng(s);
    ta=tic;
    U={};
    evalc('U = aca_nd(N*ones(1,d), Af, aca_tol);');
    tim_aca(is) = toc(ta);
    k_aca(is)   = size(U{1},2);
    err_aca(is) = aca_eval(U, subs, k_aca(is), vt, nvt);

    % (B) Cheb con seme s (selezione casuale delle fibre)
    rng(s);
    t0=tic;
    Xu=[];
    evalc('Xu = run_ml(core_dims, ncheb, intervals, pi0, en, opt, coeff_levels, fun, Qh);');
    vm = eval_tt(Xu, subs);
    err_cheb(is) = norm(vm-vt)/nvt;
    tim_cheb(is) = toc(t0);

    done(is) = true;

    % ---- salvataggio INCREMENTALE (dopo ogni seme) ------------------------
    save(outfile, 'seed_list','err_cheb','tim_cheb','err_aca','tim_aca','k_aca', ...
        'done','core_dims','ncheb','coeff_levels','nreplicas','d','nsamples', ...
        'aca_tol','sample_seed');

    appendlog(logfile, sprintf(['  seed %d/%d (rng=%d): Cheb=%.3e (%.0f s) | ' ...
        'ACA rank=%d err=%.3e (%.0f s) | %s'], is, ns_seed, s, ...
        err_cheb(is), tim_cheb(is), k_aca(is), err_aca(is), tim_aca(is), ...
        eta_str(done, tim_cheb, tim_aca, t_run)));
end

%% ---- riepilogo -----------------------------------------------------------
dof_cheb = prod(core_dims) + sum(N*core_dims);        % Tucker
dof_aca  = @(rk) rk * N * d;                          % CP: rk * sum_i N_i
appendlog(logfile, sprintf(['SUMMARY Cheb (%d seeds): median=%.3e  min=%.3e  ' ...
    'max=%.3e  spread=%.1fx  | dof=%d (fixed)  | mean time=%.0f s'], nnz(done), ...
    median(err_cheb,'omitnan'), min(err_cheb), max(err_cheb), ...
    max(err_cheb)/min(err_cheb), dof_cheb, mean(tim_cheb,'omitnan')));
appendlog(logfile, sprintf(['SUMMARY ACA  (%d seeds): median err=%.3e  min=%.3e  ' ...
    'max=%.3e  spread=%.1fx  | median rank=%.0f [%d..%d]  | dof worst=%d  | ' ...
    'mean time=%.0f s'], nnz(done), ...
    median(err_aca,'omitnan'), min(err_aca), max(err_aca), ...
    max(err_aca)/min(err_aca), median(k_aca,'omitnan'), min(k_aca), max(k_aca), ...
    dof_aca(max(k_aca)), mean(tim_aca,'omitnan')));
appendlog(logfile, sprintf('tempo totale: %.0f s (%.2f h)', toc(t_run), toc(t_run)/3600));
appendlog(logfile, 'DONE');

%% ---- Figure --------------------------------------------------------------
if make_figures
    col_cr  = [0 0.45 0.74];      % blu     = Cheb+Riemann
    col_aca = [0.85 0.33 0.10];   % arancio = ACA

    % (1) errore L2: le due distribuzioni affiancate
    f1 = figure('Visible','off','Position',[100 100 620 500]);
    hold on
    draw_box(1, err_cheb, col_cr);
    draw_box(2, err_aca,  col_aca);
    set(gca,'YScale','log'); grid on
    xlim([0.4 2.6]); xticks([1 2]);
    xticklabels({'Cheb+Riemann','ACA'});
    ylabel('relative L2 error (test set)');
    title(sprintf('Case study 1 under repair, d = %d : %d semi', d, nnz(done)));
    saveas(f1, fullfile(scriptDir, [figprefix '_err_boxes.png']));

    % (2) gradi di liberta': Cheb fisso per costruzione vs nuvola ACA
    f2 = figure('Visible','off','Position',[100 100 620 500]);
    hold on
    draw_box(1, dof_cheb*ones(nnz(done),1), col_cr);
    draw_box(2, dof_aca(k_aca),             col_aca);
    grid on
    xlim([0.4 2.6]); xticks([1 2]);
    xticklabels({'Cheb+Riemann','ACA'});
    ylabel('degrees of freedom (DoF)');
    title(sprintf('DoF: fixed rank [%s] vs adaptive ACA rank', num2str(core_dims)));
    saveas(f2, fullfile(scriptDir, [figprefix '_dof_boxes.png']));

    close([f1 f2]);
end

fprintf('\nSalvati risultati in %s\n', outfile);
end

%% ===== locali =============================================================
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

% Stima del tempo residuo dal costo medio per seme gia' osservato.
function s = eta_str(done, tim_cheb, tim_aca, t_run)
    per = mean(tim_cheb + tim_aca, 'omitnan');
    if isnan(per), s = ''; return; end
    remaining = per * nnz(~done);
    s = sprintf('elapsed %.0f min, ETA %.0f min', toc(t_run)/60, remaining/60);
end

function X = run_ml(core_dims, ncheb, intervals, pi0, en, opt, coeff, fun, Qh)
    d=numel(intervals); Xp=[];
    for lev=1:numel(ncheb)
        if lev==1, td=core_dims; else, td=[ncheb(lev), ncheb(lev)*ones(1,d-1)]; end
        vals=cell(1,d); for i=1:d, vals{i}=chebpts(td(i),intervals{i}); end
        if isempty(Xp)
            [~,X]=cheb_riemm_sparse_mixed(core_dims,td,ncheb(lev),vals,intervals,pi0,en,opt(lev),coeff(lev),true,[],fun,Qh);
        else
            [~,X]=cheb_riemm_sparse_mixed(core_dims,td,ncheb(lev),vals,intervals,pi0,en,opt(lev),coeff(lev),false,Xp,fun,Qh);
        end
        Xp=X;
    end
    X=Xp;
end

function vals=eval_tt(X,subs)
    ns=size(subs,1); dd=size(subs,2); vals=zeros(ns,1);
    for k=1:ns, cols=cell(1,dd); for m=1:dd, cols{m}=X.U{m}(subs(k,m),:).'; end
        vals(k)=double(ttv(X.core,cols,1:dd)); end
end

function e=aca_eval(U,subs,rr,vt,nvt)
    ns=size(subs,1); dd=size(subs,2); va=zeros(ns,1);
    for k=1:ns, w=ones(1,rr); for m=1:dd, w=w.*U{m}(subs(k,m),1:rr); end, va(k)=sum(w); end
    e=norm(va-vt)/nvt;
end

function f=fib_od(j,i,grid,pi0,en,fun,Qh)
    d=numel(grid); prm=cell(1,d-1); for m=2:d, prm{m-1}=grid{m}(i(m)); end
    if j==1, f=fiber_time(grid{1},prm,pi0,en,fun,Qh);
    else, Nj=numel(grid{j}); f=zeros(Nj,1);
        for a=1:Nj, prm{j-1}=grid{j}(a); v=fiber_time(grid{1},prm,pi0,en,fun,Qh); f(a)=v(i(1)); end
    end
    f=f(:);
end

function v=ur(tvec,Q,pi0,en)
    tvec=tvec(:); if max(tvec)==0, v=zeros(size(tvec)); return; end
    W=KolmogorovIntegralODE(tvec,Q,pi0); v=(en(:).'*W.').'; v=v./tvec; v(tvec==0)=0;
end

function draw_box(x, data, col)
    % box "fatto a mano" (quartili + mediana) piu' nuvola dei punti, senza
    % dipendere dallo Statistics Toolbox.
    data = data(~isnan(data));
    if isempty(data), return; end
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
