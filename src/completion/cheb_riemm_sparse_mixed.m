function [Res, Xtr, X0_out, rg0_out, norm_rg0_out] = cheb_riemm_sparse_mixed(core_dims, tensor_dims, n0, values, intervals, pi0, en, options, coefficient, base_level, F, fun, Qh)
% [MIXED-RANK] Copia di cheb_riemm_sparse con UN'UNICA differenza sostanziale:
% la varieta' e' costruita con il rango multilineare PER-MODO core_dims
% (invece di r = max(core_dims) uguale su tutti i modi). In questo modo si
% puo' usare un rango ALTO nel tempo (modo 1) e BASSO nei parametri:
% le matrici KRU nel costo/gradiente/Hessiano ESCLUDONO il modo 1, quindi
% il fattore critico r^(d-1) dipende solo dai ranghi dei parametri.
% Tutto il resto del file e' identico all'originale (che non viene toccato),
% tranne la [CACHE per punto X]: costo/gradiente/Hessiano condividono via
% store di Manopt le righe dei fattori, KRU e i residui, evitando di
% ricostruire la matrice grande a ogni chiamata nello stesso punto.
%
% Richiede che il punto iniziale abbia rango core_dims: al livello base la
% griglia deve avere tensor_dims == core_dims (es. [5, 2, 2, ..., 2]).
%
% Qh (opzionale, ultimo arg): handle del generatore inoltrato a fiber_time /
% eval_all. Se assente/vuoto si usa evalQ_extended (comportamento invariato).
if ~exist('Qh', 'var'), Qh = []; end
%ATTENZIONE: funziona bene solo sulle fibre, è scritta proprio per questa
%applicazione

% output opzionali (assegnati di default per non rompere il ramo base_level)
X0_out = []; rg0_out = []; norm_rg0_out = [];

% handle opzionale della misura (fun assente/vuoto => instantaneous)
if ~exist('fun', 'var'), fun = []; end

    %two step optimization procedure, via chebyshev interpolation and
    %riemannian tensor completion, via tensor toolbox

    if ~exist('tenrand', 'file')
        fprintf('Tensor Toolbox version 2.6 or higher is required.\n');
        return;
    end

    if ~exist('base_level')
        base_level = false;
    end

    %core dims contains the rank

    if length(tensor_dims)~=length(core_dims)
        fprintf('Missing rank information.\n');
        return;
    end

    total_entries = prod(tensor_dims);
    d = length(tensor_dims);

% numero target di osservazioni

maxfibers = prod(tensor_dims(2:end));
fiber_len = tensor_dims(1);
% numero di fibre da prendere
subs = cell(1, d-1);
if base_level
    fiber_ind = 1 : maxfibers;
    nfibers = maxfibers;
    [subs{:}] = ind2sub(tensor_dims(2:end), fiber_ind);
else
    nr = round(coefficient* (sum(core_dims .* tensor_dims) + prod(core_dims)));
    fprintf('target nr = %e, total entries = %d (%2.1f%% of total)\n', ...
        nr, total_entries, 100 * nr / total_entries);
    nfibers = ceil(nr / fiber_len);
    % numero massimo di fibre possibili
    nfibers = min(nfibers, maxfibers);

    fprintf('number of selected fibers = %d\n', nfibers);

    % scelgo fibre casuali. Con maxfibers = prod(n_k) enorme (d grande, N
    % grande) l'indice lineare supera 2^53 (limite di randperm e soglia di
    % rappresentabilita' esatta dei double), quindi randperm/ind2sub non sono
    % utilizzabili. In quel caso campiono i multi-indici DIRETTAMENTE (un
    % indice casuale per ciascun modo); con nfibers << maxfibers le fibre sono
    % quasi certamente distinte. Sotto la soglia resta il comportamento
    % originale (randperm), cosi' i risultati degli altri esperimenti non
    % cambiano.
    if maxfibers < 2^53
        fiber_ind = randperm(maxfibers, nfibers);
        [subs{:}] = ind2sub(tensor_dims(2:end), fiber_ind);
    else
        for jm = 1:d-1
            subs{jm} = randi(tensor_dims(jm+1), 1, nfibers);
        end
    end
end

fiber_subs = zeros(nfibers,d-1);
fiber_vals = cell(nfibers,1);

if base_level
    PA = tensor(zeros(tensor_dims));
end

for k = 1:nfibers

    param = cell(1,d-1);

    for j = 2:d
        val = subs{j-1}(k);
        fiber_subs(k,j-1) = val;
        param{j-1} = values{j}(val);

    end

    fiber_vals{k} = fiber_time(values{1},param,pi0,en,fun,Qh);

    if base_level
        idx = num2cell(fiber_subs(k,:));
        PA(:,idx{:}) = fiber_vals{k};
    end

end

% valori osservati come matrice n1 x nfibers (usata dalle versioni vettorizzate)
FVmat = [fiber_vals{:}];

    nobs = nfibers * tensor_dims(1);
    fprintf('effective observed entries = %d (%2.3f%% of total)\n', ...
        nobs, 100*nobs/total_entries);

    if base_level
        UList = cell(1, d);
        for j = 1 : d; UList{j} = eye(size(PA, j)); end
        Xtr = ttensor(tensor(PA), UList{:});
        Res = 0.0;
        return
    end

    %pause
    %Riemannian optimization problem

    % Pick the submanifold of tensors of size n1-by-...-by-nd of
    % multilinear rank (r1, ..., rd).
    % [MIXED-RANK] rango per-modo: qui sta l'unica differenza rispetto a
    % cheb_riemm_sparse (che usa r = max(core_dims) uguale su tutti i modi).
    problem.M = fixedranktensorembeddedfactory(tensor_dims, core_dims);

    % --- Override di M.transp (trasporto vettoriale) -----------------------
    % La versione del factory passa per il tensore ambiente PIENO
    % (ttm(ttm(G,U),U_tilde,'t') costruisce un n1 x ... x nd denso) e per d
    % grande esaurisce la memoria. Matematicamente pero' il trasporto usa
    % solo le matrici piccole A{j} = U_tilde{j}'*U{j} e B{j} = U_tilde{j}'*V{j}:
    % transport_small (in fondo al file) fa le stesse operazioni con sole
    % contrazioni piccole ed e' identico in aritmetica esatta.
    problem.M.transp = @transport_small;

    % --- Override di M.inner / M.norm (metrica STABILE) --------------------
    % Il factory calcola i termini della metrica come
    %     innerprod(C, ttm(C, V'*W, i))  =  trace((V'*W) * C_i*C_i')
    % contraendo la Gram del core (entrate ~ ||C||^2, enormi per d grande)
    % con la matrice piccola V'*W: pavimento di errore assoluto
    % ~ eps*||C||^2, che sommerge curvature/gradienti piccoli (il tCG vede
    % "curvature negative" spurie e i passi vengono rigettati).
    % Forma equivalente ma stabile: contrarre PRIMA il core con V,
    %     <C x_i V, C x_i W> = <V*C_i, W*C_i>,
    % cosi' l'errore e' proporzionale alle quantita' vere (~eps*||VC||*||WC||).
    problem.M.inner = @inner_stable;
    problem.M.norm  = @(X, eta) sqrt(max(inner_stable(X, eta, eta), 0));


    % [CACHE per punto X] costo, gradiente e Hessiano condividono tramite lo
    % store di Manopt le quantita' che dipendono SOLO dal punto: righe dei
    % fattori sulle fibre (Mu_rows), il loro Khatri-Rao (KRU, la matrice
    % grande) e i residui R. Cosi' dentro una stessa iterazione del
    % trust-region KRU viene costruita UNA volta invece che a ogni chiamata.
    function store = prep_kru(XX, store)
        if ~isfield(store, 'KRU')
            U = XX.X.U;  s = ndims(XX.X.core);
            Mu = cell(1, s-1);
            for idx = 1:(s-1)
                h = s - idx + 1;      % ordine di modo decrescente s,...,2
                Mu{idx} = U{h}(fiber_subs(:, h-1), :);
            end
            store.Mu_rows = Mu;
            store.KRU     = rowkron(Mu);   % nf x prod(r_{2..s})
        end
    end

    function store = prep_res(XX, store)
        if ~isfield(store, 'R')
            C  = XX.X.core;
            r1 = size(C, 1);
            C1 = reshape(double(C), r1, []);        % matricizzazione modo-1
            store.R = XX.X.U{1} * (C1 * store.KRU.') - FVmat;  % n1 x nf
        end
    end

    problem.cost = @cost;
    function [f, store] = cost(XX, store)
        if nargin < 2, store = struct(); end
        store = prep_kru(XX, store);
        store = prep_res(XX, store);
        f = 0.5 * sum(store.R(:).^2);
    end

%NUOVO DA QUI ------------------------------
problem.grad = @grad;
    function [rg, store] = grad(XX, store)
        if nargin < 2, store = struct(); end
        store = prep_kru(XX, store);
        store = prep_res(XX, store);
        Es.R = store.R;
        rg = rgrad(XX, Es, store.Mu_rows, store.KRU);
    end

% Scelta dell'Hessiano riemanniano (options.hessian):
%   'exact' (default) : ehess2rhess del factory con ambiente sptensor.
%                       Esatto, ma la proiezione dell'ambiente sparso
%                       densifica un unfolding n^(d-1) x r: NON scala per
%                       d grande (out of memory gia' con d = 9, n = 16).
%   'gn'              : Gauss-Newton proiettato, Proj_X(P(eta)), calcolato
%                       interamente sulle fibre riusando rgrad (che E'
%                       la proiezione tangente di un ambiente supportato
%                       sulle fibre). Trascura solo il termine di curvatura
%                       (lineare nel residuo): simmetrico e semidefinito
%                       positivo, adatto a trustregions in ogni dimensione.
if isfield(options, 'hessian') && ~isempty(options.hessian)
    hessian_kind = lower(options.hessian);
else
    hessian_kind = 'exact';
end
switch hessian_kind
    case 'exact'
        problem.hess = @hess;
    case 'gn'
        problem.hess = @hess_gn;
    otherwise
        error('options.hessian sconosciuto: %s', hessian_kind);
end

%% Calcolo del gradiente euclideo strutturato sulle fibre (vettorizzato)
function E_struct = E_fibers(XX)
    X = XX.X;  U = X.U;  C = X.core;
    s = ndims(C);

    Mu = cell(1, s-1);
    for idx = 1:(s-1)
        h = s - idx + 1;
        Mu{idx} = U{h}(fiber_subs(:, h-1), :);
    end
    KRU = rowkron(Mu);                        % nf x prod(r_{2..s})
    r1  = size(C, 1);
    C1  = reshape(double(C), r1, []);
    GX  = U{1} * (C1 * KRU.');                % n1 x nf
    R   = GX - FVmat;                         % n1 x nf  residui

    E_struct.R = R;                           % n1 x nf (residui sulle fibre)
end

% gradiente riemanniano direttamente senza egrad - VETTORIZZATO sulle fibre.
% E' la proiezione tangente del tensore ambiente supportato sulle fibre
% osservate, con valori E_struct.R (n1 x nfibers): vale sia per il gradiente
% (R = residui) sia per l'Hessiano di Gauss-Newton (R = P(eta) sulle fibre).
% Mu_rows/KRU opzionali: blocchi riga dei fattori (ordine di modo
% decrescente s,...,2, come in cost/E_fibers) e loro prodotto di Khatri-Rao
% per riga, riusabili fra piu' chiamate nello stesso punto X.
function rg = rgrad(XX, E_struct, Mu_rows, KRU)

    X  = XX.X;
    U  = X.U;
    C  = X.core;
    s  = ndims(C);
    r  = size(C);

    R = E_struct.R;                          % n1 x nf

    if nargin < 3 || isempty(Mu_rows)
        Mu_rows = cell(1, s-1);
        for idx = 1:(s-1)
            h = s - idx + 1;
            Mu_rows{idx} = U{h}(fiber_subs(:, h-1), :);
        end
    end
    if nargin < 4 || isempty(KRU)
        KRU = rowkron(Mu_rows);              % nf x prod(r_{2..s})
    end

    P1 = U{1}' * R;                          % r1 x nf

    % core: dG_(1) = sum_l (U1'*r_l) * kron(righe dei fattori) = P1 * KRU
    dG = tensor(reshape(P1 * KRU, r));

    dV = cell(1, s);

    % modo 1: dV{1} = sum_l r_l * (KRU(l,:) * Cpinv{1})
    dV{1} = R * (KRU * XX.Cpinv{1});         % n1 x r1

    % modi k >= 2: il kron esclude il modo k e include (U1'*r_l) come
    % blocco piu' interno (modo 1, il piu' veloce), coerente con
    % l'ordinamento delle colonne di tenmat(C,k) usato in Cpinv{k}
    P1t = P1.';                              % nf x r1
    for kk = 2:s
        blocks = cell(1, s-1);
        b = 0;
        for h = s:-1:2
            if h == kk, continue; end
            b = b + 1;
            blocks{b} = Mu_rows{s - h + 1};
        end
        blocks{b+1} = P1t;
        T = rowkron(blocks) * XX.Cpinv{kk};  % nf x r_kk
        Sk = sparse(fiber_subs(:, kk-1), 1:nfibers, 1, tensor_dims(kk), nfibers);
        dV{kk} = Sk * T;                     % n_kk x r_kk
    end

    % Proiezione ortogonale
    for q = 1:s
        dV{q} = dV{q} - U{q} * (U{q}' * dV{q});
    end

    rg.G = dG;
    rg.V = dV;
end


%% ===== HESSIANO ============================================================
% Tutto strutturato sulle fibre: nessun tensore pieno viene mai costruito.
% L'ambient (egrad euclideo e P(eta)) e' un sptensor supportato solo sulle
% fibre osservate; le proiezioni e il termine di curvatura sono delegati al
% factory tramite ehess2rhess (la varieta' e' curva: la curvatura NON e' nulla).

% Gradiente euclideo come tensore ambient SPARSO (solo fibre osservate):
%   E = P(X) - b ,  con b = fiber_vals.
function E = egrad_sparse(XX)
    Es = E_fibers(XX);                       % R: n1 x nf residui sulle fibre
    E  = fiber_sptensor(Es.R);               % sptensor supportato sulle fibre
end

% Hessiano euclideo lungo eta:  ehess[eta] = P(eta), ambient SPARSO.
% Sulla fibra (.,j2,...,js):
%   P(eta) = U1*( G x_{h>=2} U_h(j_h,:) + sum_{k>=2} C x_{h!=k} U_h(j_h,:) x_k V_k(j_k,:) )
%          + V1*( C x_{h>=2} U_h(j_h,:) )
function H = ehess_sparse(XX, eta)
    H = fiber_sptensor(ehess_fib(XX, eta));
end

% Valori di P(eta) sulle fibre osservate (matrice n1 x nfibers).
% Au/KRU opzionali: righe dei fattori e loro Khatri-Rao, riusabili fra
% chiamate nello stesso punto X (le righe di V dipendono da eta e vengono
% sempre ricostruite).
function Hfib = ehess_fib(XX, eta, Au, KRU)
    X = XX.X;  U = X.U;  C = X.core;  s = ndims(C);
    G = eta.G;  V = eta.V;
    r1 = size(C, 1);
    C1 = reshape(double(C), r1, []);          % matricizzazione modo-1 del core
    G1 = reshape(double(G), r1, []);          % idem per la variazione del core

    % righe selezionate (fattori U e variazioni V) per ogni fibra, modi 2..s
    if nargin < 3 || isempty(Au)
        Au = cell(1, s-1);   % U-rows
        for idx = 1:(s-1)
            h = s - idx + 1;                  % ordine decrescente s,...,2
            Au{idx} = U{h}(fiber_subs(:, h-1), :);
        end
    end
    Av = cell(1, s-1);       % V-rows
    for idx = 1:(s-1)
        h = s - idx + 1;
        Av{idx} = V{h}(fiber_subs(:, h-1), :);
    end

    if nargin < 4 || isempty(KRU)
        KRU = rowkron(Au);                    % nf x prod(r_{2..s})
    end
    GC  = C1 * KRU.';                         % r1 x nf  (per il termine V1*gC)

    acc = G1 * KRU.';                         % r1 x nf  (A: variazione del core)
    for k = 2:s                               % (B) variazioni dei fattori
        Mk = Au;
        Mk{s - k + 1} = Av{s - k + 1};        % modo k: riga V al posto di U
        acc = acc + C1 * rowkron(Mk).';       % r1 x nf
    end

    Hfib = U{1}*acc + V{1}*GC;                % n1 x nf  = P(eta) sulle fibre
end

% Hessiano riemanniano di GAUSS-NEWTON: Hrh = Proj_X(P(eta)), tutto sulle
% fibre. rgrad(XX, E_struct) calcola esattamente la proiezione tangente di
% un tensore ambiente supportato sulle fibre osservate (con i valori in
% E_struct.R), quindi basta passargli i valori di P(eta).
% Il termine di curvatura (Weingarten), lineare nel residuo, e' trascurato:
% l'operatore resta simmetrico e semidefinito positivo.
function [Hrh, store] = hess_gn(XX, eta, store)
    if nargin < 3, store = struct(); end
    % [CACHE] riusa Mu_rows/KRU del punto (condivisi con costo e gradiente)
    store = prep_kru(XX, store);
    Hs.R = ehess_fib(XX, eta, store.Mu_rows, store.KRU);
    Hrh  = rgrad(XX, Hs, store.Mu_rows, store.KRU);
end

% Hessiano riemanniano = Proj_X(ehess[eta]) + curvatura(egrad,eta).
% Il termine di curvatura (Weingarten) e' gestito dal factory: gli passiamo
% egrad ed ehess come ambient sparsi e otteniamo direttamente rhess.
function [Hrh, store] = hess(XX, eta, store)
    if nargin < 3 || ~isfield(store, 'Eamb')
        store.Eamb = egrad_sparse(XX);        % egrad sparso, una volta per punto
    end
    Heh = ehess_sparse(XX, eta);              % P(eta) sparso
    Hrh = problem.M.ehess2rhess(XX, store.Eamb, Heh, eta);
end

%% --- Helper vettorizzati (niente ttv nei cicli sulle fibre) ----------------

% Prodotto di Khatri-Rao "riga-per-riga": date le matrici M{1},...,M{m}
% (ognuna nf x r_t), restituisce K (nf x prod(r_t)) con
%   K(l,:) = kron( M{1}(l,:), M{2}(l,:), ..., M{m}(l,:) ).
% M{1} e' il blocco piu' esterno (lento), M{end} il piu' interno (veloce):
% coerente con la matricizzazione modo-1 reshape(core, r1, []).
function K = rowkron(M)
    K = M{1};
    for t = 2:numel(M)
        Q  = M{t};
        nf = size(K, 1);
        p  = size(K, 2);
        q  = size(Q, 2);
        % index = iq + (ip-1)*q  ->  Q (interno) varia piu' velocemente
        K = reshape(reshape(Q, nf, q, 1) .* reshape(K, nf, 1, p), nf, p*q);
    end
end

% Costruisce un sptensor supportato sulle fibre osservate, dati i valori
% Vfib (n1 x nfibers): colonna l = valori lungo il modo 1 della fibra l.
function S = fiber_sptensor(Vfib)
    n1 = tensor_dims(1);
    col1 = repmat((1:n1)', nfibers, 1);            % indice di modo 1
    rest = repelem(fiber_subs, n1, 1);             % indici dei modi 2..d
    S = sptensor([col1, rest], Vfib(:), tensor_dims);
end

% Prodotto interno sullo spazio tangente: identico in aritmetica esatta a
% M.inner del factory, ma calcolato nella forma stabile <V*C_i, W*C_i>
% (vedi commento all'override sopra).
function ip = inner_stable(X, eta, zeta)
    C = X.X.core;
    s = ndims(C);
    ip = innerprod(eta.G, zeta.G);
    for ii = 1:s
        Ci = double(tenmat(C, ii));       % r x prod(r_altri)
        Mv = eta.V{ii}  * Ci;             % n_ii x prod(r_altri)
        Mw = zeta.V{ii} * Ci;
        ip = ip + Mv(:)' * Mw(:);
    end
end

% Trasporto vettoriale da X a Y: stessa matematica di M.transp del factory
% (proiezione ortogonale, Kressner et al.), ma senza mai costruire il
% tensore ambiente pieno. Servono solo le matrici piccole
%   A{j} = U_tilde{j}'*U{j},  B{j} = U_tilde{j}'*V{j}   (r x r)
% e contrazioni del core (dimensione massima r^(d-1) x n_i).
function eta_t = transport_small(X, Y, xi)
    C  = X.X.core;   U  = X.X.U;   Ut = Y.X.U;
    G  = xi.G;       V  = xi.V;
    s  = ndims(C);

    A = cell(1, s);  B = cell(1, s);
    for j = 1:s
        A{j} = Ut{j}' * U{j};
        B{j} = Ut{j}' * V{j};
    end

    % componente core: ttm(ttm(G,U),Ut,'t') = G x_j A{j}, e analoghi con V
    Gt = ttm(G, A);
    for i = 1:s
        Mi = A;  Mi{i} = B{i};
        Gt = Gt + ttm(C, Mi);
    end

    % componenti dei fattori
    Vt = cell(1, s);
    for i = 1:s
        modesWoI = [1:i-1, i+1:s];

        % ttm(ttm(G,U), UtWoI, modesWoI, 't') = G x_i U{i} x_{j~=i} A{j}
        T = ttm(ttm(G, U{i}, i), A(modesWoI), modesWoI);
        % termine k = i: C x_i V{i} x_{j~=i} A{j}
        T = T + ttm(ttm(C, V{i}, i), A(modesWoI), modesWoI);
        % termini k ~= i: C x_i U{i} x_k B{k} x_{j~=i,k} A{j}
        for k = modesWoI
            Mk = A;  Mk{k} = B{k};
            T = T + ttm(ttm(C, U{i}, i), Mk(modesWoI), modesWoI);
        end

        bp = double(tenmat(T, i)) * Y.Cpinv{i};
        Vt{i} = bp - Ut{i} * (Ut{i}' * bp);
    end

    eta_t.G = Gt;
    eta_t.V = Vt;
end


%% ===== SELFTEST brute-force (casi piccoli) ================================
% Verifica l'implementazione REALE (le stesse funzioni annidate usate
% dall'ottimizzatore) contro riferimenti indipendenti:
%   1) costo   vs 0.5*||P_Omega(full(X0)) - P_Omega(A)||^2 sul tensore pieno
%   2) rgrad   vs proiezione tangente del factory (egrad2rgrad/proj)
%              dell'ambiente denso dei residui
%   3) hess_gn vs Proj( maschera_Omega( tangent2ambient(eta) ) ) di riferimento
%   4) simmetria e semidefinitezza positiva dell'operatore GN
%   5) coerenza della cache (store condiviso vs store nuovi)
function selftest_impl(X0)
    fprintf('\n--- SELFTEST implementation (brute force on the full tensor) ---\n');
    M_ = problem.M;
    if isfield(M_, 'egrad2rgrad'), projfun = @(x, a) M_.egrad2rgrad(x, a);
    else,                          projfun = @(x, a) M_.proj(x, a);
    end
    amb = @(x, e) full(M_.tangent2ambient(x, e));   % tangente -> tensore pieno

    % 1) costo
    Xfull = double(full(X0.X));
    Rbf = zeros(tensor_dims(1), nfibers);
    for l = 1:nfibers
        idx = num2cell(fiber_subs(l, :));
        Rbf(:, l) = Xfull(:, idx{:}) - FVmat(:, l);
    end
    f_bf = 0.5 * sum(Rbf(:).^2);
    [f_impl, st_] = cost(X0, struct());
    fprintf('1) costo:    |impl - brute| / max(1,|brute|)   = %.3e\n', ...
        abs(f_impl - f_bf) / max(1, abs(f_bf)));

    % 2) gradiente
    Eamb  = full(fiber_sptensor(Rbf));
    g_ref = projfun(X0, Eamb);
    [g_impl, st_] = grad(X0, st_);
    dg  = norm(amb(X0, g_impl) - amb(X0, g_ref));
    ng  = norm(amb(X0, g_ref));
    fprintf('2) gradiente: ||impl - rif|| / ||rif||          = %.3e\n', dg / max(ng, eps));

    % 3) Hessiano GN lungo una direzione tangente casuale
    rng(0);
    eta  = M_.randvec(X0);
    Aeta = full(M_.tangent2ambient(X0, eta));
    Amask = tensor(zeros(tensor_dims));
    for l = 1:nfibers
        idx = num2cell(fiber_subs(l, :));
        Amask(:, idx{:}) = Aeta(:, idx{:});
    end
    h_ref  = projfun(X0, Amask);
    h_impl = hess_gn(X0, eta);
    dh = norm(amb(X0, h_impl) - amb(X0, h_ref));
    nh = norm(amb(X0, h_ref));
    fprintf('3) hess GN:  ||impl - rif|| / ||rif||           = %.3e\n', dh / max(nh, eps));

    % 4) simmetria e PSD dell'operatore GN
    e1 = M_.randvec(X0);  e2 = M_.randvec(X0);
    s12 = inner_stable(X0, e1, hess_gn(X0, e2));
    s21 = inner_stable(X0, hess_gn(X0, e1), e2);
    qf  = inner_stable(X0, e1, hess_gn(X0, e1));
    fprintf('4) hess GN:  |<e1,He2>-<He1,e2>| = %.3e | <e,He> = %.3e (atteso >= 0)\n', ...
        abs(s12 - s21), qf);

    % 5) coerenza della cache: store condiviso vs store nuovi
    f_shared = cost(X0, st_);
    g_fresh  = grad(X0, struct());
    dgc = norm(amb(X0, g_fresh) - amb(X0, g_impl));
    fprintf('5) cache:    |costo(store cond.) - costo| = %.3e | ||grad(fresh)-grad(cond.)|| = %.3e\n', ...
        abs(f_shared - f_impl), dgc);

    % 6) TEST DECISIVO sulla tecnica Khatri-Rao dell'Hessiano: con RESIDUO
    % NULLO (dati = valori esatti di X0 sulle fibre) X0 e' un punto critico
    % esatto e l'Hessiano GN coincide con l'Hessiano VERO del costo: il
    % test di Taylor di Manopt deve dare slope ~3 (valido anche con
    % retrazione del 1o ordine, perche' il punto e' critico). Se la
    % costruzione via Khatri-Rao fosse sbagliata, lo slope 3 non uscirebbe.
    FVmat_orig = FVmat;
    GXfib = zeros(tensor_dims(1), nfibers);
    for l = 1:nfibers
        idx = num2cell(fiber_subs(l, :));
        GXfib(:, l) = Xfull(:, idx{:});
    end
    FVmat = GXfib;                       % residuo esattamente nullo
    g0 = grad(X0, struct());
    fprintf(['6) residuo nullo: ||grad|| = %.3e (atteso ~0); ', ...
             'checkhessian sotto: SLOPE ATTESO ~3\n'], ...
        problem.M.norm(X0, g0));
    checkhessian(problem, X0); drawnow;

    % 7) GN vs ''exact'' (ehess2rhess con termine di Weingarten): la
    % differenza dei due Hessiani e' il termine di curvatura, LINEARE nel
    % residuo: scalando il residuo di 100 deve scalare di ~100.
    rng(3);
    Rdir = randn(size(FVmat));
    dHs = zeros(1, 2); ss = [1e-1, 1e-3];
    for q7 = 1:2
        FVmat = GXfib - ss(q7) * Rdir;
        hg = hess_gn(X0, eta);
        he = hess(X0, eta, struct());
        dv = norm(amb(X0, he) - amb(X0, hg));
        dHs(q7) = dv;
    end
    fprintf(['7) ||H_exact - H_gn||: s=1e-1 -> %.3e | s=1e-3 -> %.3e | ', ...
             'rapporto = %.1f (atteso ~100)\n'], dHs(1), dHs(2), dHs(1)/dHs(2));
    FVmat = FVmat_orig;                  % ripristino dei dati veri

    fprintf('--- fine SELFTEST (attesi: 1-3,5 ~1e-14; 4 PSD; 6 slope 3; 7 rapporto ~100) ---\n\n');
end

%NUOVO FINO A QUI ----------------------------

    %SELECTION OF THE STARTING POINT FOR THE RIEMANNIAN OPTIMIZATION
    if ~exist('F', 'var') || isempty(F)
        d_start = length(intervals);
        values_start = cell(1,d_start);
        for i = 1:d_start
            values_start{i} = chebpts(n0, intervals{i});
        end
        fprintf('F not provided, computing it ...\n');
        %pause
        F = eval_all(values_start, pi0, en, fun, Qh); %full small tensor
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

    % --- SELFTEST brute-force (solo casi piccoli) ---------------------------
    % Attivalo con options.selftest = true: confronta costo, gradiente e
    % Hessiano GN dell'implementazione (con cache e ranghi misti) contro
    % riferimenti INDIPENDENTI calcolati sul tensore pieno con le proiezioni
    % del factory di Manopt. Tutto al punto X0.
    if isfield(options, 'selftest') && options.selftest
        if total_entries > 1e6
            warning('selftest skipped: tensor too large (%d entries).', total_entries);
        else
            selftest_impl(X0);
        end
    end

    % --- Validazione opzionale di gradiente e Hessiano (al punto X0) ---
    % Attivala passando options.check_derivatives = true.
    % Atteso: checkgradient slope ~2, checkhessian slope ~3 e simmetria ~1e-12.
    if isfield(options, 'check_derivatives') && options.check_derivatives
        fprintf('--- checkgradient (slope atteso ~2) ---\n');
        checkgradient(problem, X0);
        fprintf('--- checkhessian (slope atteso ~3, simmetria ~1e-12) ---\n');
        checkhessian(problem, X0);
        drawnow;
    end

    % --- Modalita' "solo gradiente": salta l'ottimizzazione --------------
    % Con options.grad_only = true non si ottimizza: Xtr = X0 e a valle si
    % calcola solo il gradiente riemanniano in X0 (per controlli esterni).
    if isfield(options, 'grad_only') && options.grad_only
        Xtr = X0;
    else
        % [CACHE] limita la profondita' dello storedb di Manopt: ogni store
        % puo' contenere KRU (grande, fino a ~GB); al trust-region bastano
        % pochi punti vivi. Override con options.storedepth.
        if ~isfield(options, 'storedepth') || isempty(options.storedepth)
            options.storedepth = 3;
        end

        % --- Scelta del solver: default trust-regions (usa l'Hessiano) ---
        % Override con options.solver = 'conjugategradient' | 'steepestdescent'.
        if isfield(options, 'solver') && ~isempty(options.solver)
            solver = lower(options.solver);
        else
            solver = 'trustregions';
        end
        switch solver
            case 'trustregions'
                Xtr = trustregions(problem, X0, options);
            case 'rlbfgs'
                % L-BFGS Riemanniano: solo gradienti, niente Hessiano
                if ~exist('rlbfgs', 'file')
                    error('rlbfgs not found: a Manopt version including it is required.');
                end
                Xtr = rlbfgs(problem, X0, options);
            case 'conjugategradient'
                Xtr = conjugategradient(problem, X0, options);
            case 'steepestdescent'
                Xtr = steepestdescent(problem, X0, options);
            otherwise
                error('solver sconosciuto: %s', solver);
        end
    end

    % verifica gradiente riemanniano = 0 al punto di convergenza
    rg_final = problem.grad(Xtr);
    norm_rg   = problem.M.norm(Xtr, rg_final);

    rg_init  = problem.grad(X0);
    norm_rg0 = problem.M.norm(X0, rg_init);
    fprintf('||rgrad f(X0)||  = %e\n', norm_rg0);
    fprintf('||rgrad f(Xtr)|| = %e\n', norm_rg);
    fprintf('riduzione: %e\n', norm_rg / norm_rg0);

    % Output opzionali: punto iniziale, gradiente riemanniano in X0 e sua norma
    % (utili per controlli esterni; X0 e' il punto esteso via cheb_approx).
    X0_out      = X0;
    rg0_out     = rg_init;
    norm_rg0_out = norm_rg0;

    % Assegnazione finale
    Xtr = Xtr.X;

    Res = 0.0;
end
