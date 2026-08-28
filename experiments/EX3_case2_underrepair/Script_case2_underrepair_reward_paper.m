function Script_case2_underrepair_reward_paper(cfg)
%SCRIPT_CASE2_UNDERREPAIR_REWARD_PAPER  Case study 2 (IPS) under repair.
%   Mean measure m(t) = (en . b(t)) / t, with b(t) the integral of pi over
%   [0, t], comparing mixed-rank Chebyshev + Riemannian completion against
%   ACA. The reward selects the states with a single operational unit.
%
%   PAR_ORDER lists the parameter indices in activation order: the first
%   d-1 are free, the others are frozen at the bound chosen by FIX_AT,
%   either 'mid' or 'lower'.
%
%   Writes case2_underrepair_paper_d<d>_ne<ne>.mat unless OUTFILE is given.

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

%% ---- config --------------------------------------------------------------
ne           = g('ne', 30);
d            = g('d', 5);
ncheb        = g('ncheb', [16 16]);        % base = core, poi livello finale sparso
coeff_levels = g('coeff_levels', [1 6]);
core_list    = g('core_list', { [16 4 4 4 4], [16 5 5 5 5], [16 6 6 6 6] });
nsamples     = g('nsamples', 2000);
aca_tol      = g('aca_tol', 1e-6);
sample_seed  = g('sample_seed', 0);
run_seed     = g('run_seed', 1);
max_frac     = g('max_frac', 0.50);        % soglia di onesta' sul TOTALE
do_aca       = g('do_aca', true);
resume       = g('resume', true);          % riprende da outfile se compatibile

maxiter      = g('maxiter', 120);
maxinner     = g('maxinner', 50);
tolgradnorm  = g('tolgradnorm', 1e-11);

outfile = g('outfile', sprintf('case2_underrepair_paper_d%d_ne%d.mat', d, ne));
logfile = g('logfile', sprintf('case2_underrepair_paper_d%d_ne%d_progress.txt', d, ne));
if ~isAbsPath(outfile), outfile = fullfile(here, outfile); end
if ~isAbsPath(logfile), logfile = fullfile(here, logfile); end

N = ncheb(end);
if numel(coeff_levels) ~= numel(ncheb)
    error('coeff_levels must have the same length as ncheb');
end

% 'a' e non 'w': con resume attivo il log della sessione precedente va tenuto
if resume && isfile(outfile), lmode = 'a'; else, lmode = 'w'; end
fid=fopen(logfile,lmode); fprintf(fid,'start %s\n', datestr(now)); fclose(fid);

%% ---- Modello IPS ---------------------------------------------------------
nstates = 3 + 2*ne;
tf      = 24*365*10;                       % 10 anni
pi0 = zeros(nstates,1); pi0(1) = 1;

% REWARD DEL PAPER: una sola unita' operativa, con G Up oppure Down
en = zeros(nstates,1);
en(2)           = 1;                       % (Up,1)
en(2+ne+(1:ne)) = 1;                       % (Down,1), tutte le fasi di Erlang

iv = { [1,3], [1e-4,1e-3], [1e-5,1e-4], [1e-5,1e-4], [1e-6,1e-5], [0.5,2.5], ...
       [0.25,0.75], [1e-6,1e-5], [1e-7,1e-5], [1e-6,1e-5], [1e-6,1e-5], [1e-7,1e-6] };
mid = cellfun(@(x) 0.5*(x(1)+x(2)), iv);

% ---- selezione dei parametri liberi ---------------------------------------
% par_order elenca gli indici di iv nell'ordine di attivazione: i primi p = d-1
% sono LIBERI (e diventano, in quest'ordine, le dimensioni 2..d del tensore),
% il resto e' fissato. Gli indici di iv corrispondono a par_2..par_13.
%   default            = [1 2 6 7 ...]  -> L, lambda, mu, mu_u   (under-repair)
%   ordine reliability = [3 4 5 7 8 9 10 11 12 6 1 2] -> lambda_r, lambda_b,
%                        lambda_c, mu_u, ...           ("deboli prima")
% fix_at sceglie dove bloccare i parametri non attivi:
%   'mid'   punto medio dell'intervallo  (convenzione degli script under-repair)
%   'lower' bordo inferiore              (convenzione degli script reliability)
par_order = g('par_order', [1 2 6 7 3 4 5 8 9 10 11 12]);
fix_at    = g('fix_at', 'mid');

p      = d - 1;
allpar = iv(par_order);
switch lower(fix_at)
    case 'mid',   fixv = cellfun(@(x) 0.5*(x(1)+x(2)), allpar(p+1:end));
    case 'lower', fixv = cellfun(@(x) x(1),            allpar(p+1:end));
    otherwise,    error('fix_at must be ''mid'' or ''lower''');
end
fixc = num2cell(fixv);
Qh   = @(varargin) evalQ_perm(ne, par_order, varargin{:}, fixc{:});

tmap      = @(s) s;                        % TEMPO FISICO (identita')
intervals = [ {[0, tf]}, allpar(1:p) ];

midfree = num2cell(cellfun(@(x) 0.5*(x(1)+x(2)), allpar(1:p)));
s0  = max(abs(ur(tmap(chebpts(N,[0,tf])), Qh(midfree{:}), pi0, en)));
fun = @(s,Q,p0,e) ur(tmap(s),Q,p0,e)/s0;

appendlog(logfile, sprintf('free parameters (dim 2..%d): iv[%s] | fixed at ''%s''', ...
    d, num2str(par_order(1:p)), fix_at));

appendlog(logfile, sprintf(['IPS ne=%d nstates=%d d=%d PHYSICAL TIME | reward: %d ' ...
    'rewarded states out of %d (one operational unit) | s0=%.3e'], ...
    ne, nstates, d, nnz(en), nstates, s0));

%% ---- onesta' del completamento: entrate osservate / tensore finale --------
tot_final = N^d;
appendlog(logfile, sprintf('final tensor: %d^%d = %.3e entries', N, d, tot_final));
keep = true(1, numel(core_list));
obs_tot = zeros(1, numel(core_list));
for ci = 1:numel(core_list)
    cd_ = core_list{ci};
    if any(cd_ > min(ncheb(2:end)))
        appendlog(logfile, sprintf('core=[%s] DISCARDED: rank > grid of levels >=2', num2str(cd_)));
        keep(ci) = false; continue
    end
    [tot, per_lev] = observed_entries(cd_, ncheb, coeff_levels, d);
    obs_tot(ci) = tot;
    parts = strjoin(arrayfun(@(L) sprintf('lev%d=%.0f (%.2f%%)', L, per_lev(L), ...
        100*per_lev(L)/tot_final), 1:numel(per_lev), 'UniformOutput', false), '  ');
    appendlog(logfile, sprintf('core=[%s]  %s  |  TOTAL %.0f = %.2f%% of the final tensor', ...
        num2str(cd_), parts, tot, 100*tot/tot_final));
    if tot/tot_final > max_frac
        appendlog(logfile, sprintf(['   -> DISCARDED: %.1f%% exceeds max_frac=%.0f%%, ' ...
            'this would not be a true completion'], 100*tot/tot_final, 100*max_frac));
        keep(ci) = false;
    end
end
core_list = core_list(keep);  obs_tot = obs_tot(keep);
if isempty(core_list), error('no honest configuration with these parameters'); end

%% ---- griglia finale + test set esatto ------------------------------------
grd = cell(1,d); for i=1:d, grd{i}=chebpts(N,intervals{i}); end
rng(sample_seed);
subs = zeros(nsamples,d); for j=1:d, subs(:,j)=randi(N,nsamples,1); end
t0=tic;
vt = zeros(nsamples,1);
for k=1:nsamples
    prm=cell(1,d-1); for j=2:d, prm{j-1}=grd{j}(subs(k,j)); end
    v=fiber_time(grd{1},prm,pi0,en,fun,Qh); vt(k)=v(subs(k,1));
end
nvt = norm(vt);
appendlog(logfile, sprintf('test set esatto su %d punti: %.0f s (nvt=%.3e)', ...
    nsamples, toc(t0), nvt));

%% ---- ACA on-demand (riferimento) -----------------------------------------
kACA = NaN; err_aca_full = NaN; time_aca = NaN; obs_aca = NaN;
aca_nft = 0;                       % condiviso con la nested fib_od_counted
nc = numel(core_list);
res = nan(nc, 6);                  % [r_tempo r_par err tempo_s obs_tot frac_pct]
done_core = false(1, nc);
aca_done  = false;

% ---- RESUME: ACA e le configurazioni gia' fatte non vengono ricalcolate ----
% Serve perche' un run lungo puo' essere interrotto (in un caso Windows ha
% ucciso MATLAB come "applicazione bloccata" dopo 11 minuti di ACA, facendo
% perdere tutto: ACA girava per prima e veniva salvata solo dentro lo sweep).
if resume && isfile(outfile)
    S = load(outfile);
    okcfg = S.ne == ne && S.d == d && S.N == N && S.nsamples == nsamples && ...
            isequal(S.ncheb(:).', ncheb(:).') && ...
            isequal(S.coeff_levels(:).', coeff_levels(:).') && ...
            numel(S.core_list) == nc && ...
            all(cellfun(@(a,b) isequal(a,b), S.core_list(:).', core_list(:).'));
    if ~okcfg
        error(['resume=true but %s was produced with a different cfg. ' ...
               'Delete it or change outfile.'], outfile);
    end
    kACA=S.kACA; err_aca_full=S.err_aca_full; time_aca=S.time_aca; obs_aca=S.obs_aca;
    res=S.res; done_core=S.done_core; aca_done=S.aca_done;
    appendlog(logfile, sprintf('RESUME: ACA %s, %d/%d configurazioni gia'' fatte.', ...
        tern(aca_done,'gia'' fatta','da fare'), nnz(done_core), nc));
end

if do_aca && ~aca_done
    Af = @fib_od_counted;
    rng(run_seed);
    t0=tic; U={};
    evalc('U = aca_nd(N*ones(1,d), Af, aca_tol);');
    time_aca = toc(t0);
    kACA = size(U{1},2);
    obs_aca = aca_nft;
    err_aca_full = aca_eval(U, subs, kACA, vt, nvt);
    aca_done = true;
    save_partial();                % salva SUBITO: ACA e' costosa e va prima
    appendlog(logfile, sprintf('ACA: k=%d  err=%.3e  (%.0f s)  observed entries=%.0f (%.2f%%)%s', ...
        kACA, err_aca_full, time_aca, obs_aca, 100*obs_aca/tot_final, ...
        repmat('  <-- CAP maxit=1000: NOT converged', 1, kACA>=1000)));
end

%% ---- sweep sul rango parametrico -----------------------------------------
for ci = 1:nc
    if done_core(ci)
        appendlog(logfile, sprintf('core=[%s] gia'' fatto (err=%.3e, %.0f s), salto', ...
            num2str(core_list{ci}), res(ci,3), res(ci,4)));
        continue
    end
    core_dims = core_list{ci};
    clear opt
    for j=1:numel(ncheb)
        opt(j).maxiter=maxiter; opt(j).maxinner=maxinner; opt(j).tolgradnorm=tolgradnorm;
        opt(j).minstepsize=1e-15; opt(j).check_derivatives=false;
        opt(j).solver='trustregions'; opt(j).hessian='gn'; opt(j).storedepth=3;
    end
    rng(run_seed);
    t0=tic; Xu=[];
    evalc('Xu = run_ml(core_dims, ncheb, intervals, pi0, en, opt, coeff_levels, fun, Qh);');
    vm = eval_tt(Xu, subs);
    err_cheb = norm(vm-vt)/nvt;
    el = toc(t0);
    res(ci,:) = [core_dims(1) core_dims(2) err_cheb el obs_tot(ci) 100*obs_tot(ci)/tot_final];
    done_core(ci) = true;
    save_partial();
    appendlog(logfile, sprintf(['core=[%s]  Cheb=%.3e  (%.0f s)  osservate %.2f%%  |  ' ...
        'ACA=%.3e (k=%d, %.0f s)'], num2str(core_dims), err_cheb, el, ...
        100*obs_tot(ci)/tot_final, err_aca_full, kACA, time_aca));
end
appendlog(logfile, 'DONE');

%% ===== nested: salvataggio (condivide tutto lo scope) =====================
    function save_partial()
        save(outfile, 'res','core_list','ncheb','coeff_levels','kACA','err_aca_full', ...
             'time_aca','obs_aca','s0','ne','nstates','d','N','intervals','en', ...
             'nsamples','sample_seed','run_seed','maxiter','tolgradnorm','tot_final', ...
             'done_core','aca_done','obs_tot','par_order','fix_at');
    end

%% ===== nested: oracolo delle fibre per ACA, con contatore ==================
    function f = fib_od_counted(j, i)
        prm = cell(1, d-1);
        for m = 2:d, prm{m-1} = grd{m}(i(m)); end
        if j == 1
            f = fiber_time(grd{1}, prm, pi0, en, fun, Qh);
        else
            Nj = numel(grd{j}); f = zeros(Nj,1);
            for a = 1:Nj
                prm{j-1} = grd{j}(a);
                v = fiber_time(grd{1}, prm, pi0, en, fun, Qh); f(a) = v(i(1));
            end
        end
        aca_nft = aca_nft + numel(grd{j});   % entrate del tensore valutate
        f = f(:);
    end
end

%% ===== locali =============================================================
function v = getdef(cfg,f,default), if isfield(cfg,f), v=cfg.(f); else, v=default; end, end

function tf = isAbsPath(p)
    tf = ~isempty(regexp(p, '^([A-Za-z]:[\\/]|[\\/])', 'once'));
end

function appendlog(f,msg)
    fid=fopen(f,'a'); fprintf(fid,'%s  %s\n', datestr(now,'HH:MM:SS'), msg); fclose(fid);
    fprintf('%s\n', msg);
end

% Entrate del tensore osservate dalla pipeline multilivello, per livello e in
% totale. Replica la logica di cheb_riemm_sparse_mixed (righe ~52-70).
function [tot, per_lev] = observed_entries(core_dims, ncheb, coeff_levels, d)
    per_lev = zeros(1, numel(ncheb));
    per_lev(1) = prod(core_dims);                     % livello base: griglia piena
    for lev = 2:numel(ncheb)
        dims      = ncheb(lev)*ones(1,d);
        maxfibers = prod(dims(2:end));
        nr        = round(coeff_levels(lev) * (sum(core_dims .* dims) + prod(core_dims)));
        nfibers   = min(ceil(nr / dims(1)), maxfibers);
        per_lev(lev) = nfibers * dims(1);
    end
    tot = sum(per_lev);
end

function X = run_ml(core_dims, ncheb, intervals, pi0, en, opt, coeff, fun, Qh)
    d=numel(intervals); Xp=[];
    for lev=1:numel(ncheb)
        if lev==1, td=core_dims; else, td=ncheb(lev)*ones(1,d); end
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

function Q = evalQ_perm(ne, par_order, varargin)
    v = zeros(1, numel(par_order)); v(par_order) = [varargin{:}];
    vc = num2cell(v); Q = evalQ_ips(ne, vc{:});
end

function v=ur(tvec,Q,pi0,en)
    tvec=tvec(:); if max(tvec)==0, v=zeros(size(tvec)); return; end
    W=KolmogorovIntegralODE(tvec,Q,pi0); v=(en(:).'*W.').'; v=v./tvec; v(tvec==0)=0;
end

function s = tern(c,a,b), if c, s=a; else, s=b; end, end
