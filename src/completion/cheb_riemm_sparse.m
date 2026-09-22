function [Res, Xtr, X0_out, rg0_out, norm_rg0_out] = cheb_riemm_sparse(core_dims, tensor_dims, n0, values, intervals, pi0, en, options, coefficient, base_level, F, fun, Qh)
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

    % fibers
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

% valori osservati come matrice n1 x nfibers
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
    r = max(core_dims);
    problem.M = fixedranktensorembeddedfactory(tensor_dims, r*ones(1,d));
    % Simplified version with equal ranks

    % --- Override di M.transp (trasporto vettoriale)
    problem.M.transp = @transport_small;

    % --- Override di M.inner / M.norm (metrica stabile)
    problem.M.inner = @inner_stable;
    problem.M.norm  = @(X, eta) sqrt(max(inner_stable(X, eta, eta), 0));
    
    
    %faccio le versioni senza usare store
    problem.cost = @cost;
    function f = cost(XX)
        X = XX.X;  U = X.U;  C = X.core;
        s = ndims(C);
        % righe selezionate dei fattori per ogni fibra, in ordine di modo
        % decrescente s,...,2 (per il prodotto di Khatri-Rao riga-per-riga)
        Mu = cell(1, s-1);
        for idx = 1:(s-1)
            h = s - idx + 1;
            Mu{idx} = U{h}(fiber_subs(:, h-1), :);
        end
        KRU = rowkron(Mu);                  % nf x prod(r_{2..s})
        r1  = size(C, 1);
        C1  = reshape(double(C), r1, []);   % matricizzazione modo-1 del core
        GX  = U{1} * (C1 * KRU.');          % n1 x nf  valori del modello
        R   = GX - FVmat;                   % residui sulle fibre
        f   = 0.5 * sum(R(:).^2);
    end

%NUOVO DA QUI ------------------------------
problem.grad = @(X) rgrad(X, E_fibers(X));

% Scelta dell'Hessiano riemanniano (options.hessian):
%   'exact' (default) : ehess2rhess del factory con ambiente sptensor.
%                       
%   'gn'              : Gauss-Newton proiettato, Proj_X(P(eta)), calcolato
%                       interamente sulle fibre riusando rgrad . Trascura solo il termine di curvatura
%                     
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

    % modi k >= 2
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
%

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

% Valori di P(eta) sulle fibre osservate (matrice n1 x nfibers). Projections
function Hfib = ehess_fib(XX, eta, Au, KRU)
    X = XX.X;  U = X.U;  C = X.core;  s = ndims(C);
    G = eta.G;  V = eta.V;
    r1 = size(C, 1);
    C1 = reshape(double(C), r1, []);          % matricizzazione modo-1 del core
    G1 = reshape(double(G), r1, []);          % for core

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
% fibre
function [Hrh, store] = hess_gn(XX, eta, store)
    if nargin < 3, store = struct(); end
    %
    if ~isfield(store, 'Mu_rows')
        s_ = ndims(XX.X.core);
        Mu_ = cell(1, s_-1);
        for idx_ = 1:(s_-1)
            h_ = s_ - idx_ + 1;
            Mu_{idx_} = XX.X.U{h_}(fiber_subs(:, h_-1), :);
        end
        store.Mu_rows = Mu_;
        store.KRU     = rowkron(Mu_);
    end
    Hs.R = ehess_fib(XX, eta, store.Mu_rows, store.KRU);
    Hrh  = rgrad(XX, Hs, store.Mu_rows, store.KRU);
end

% Hessiano riemanniano = Proj_X(ehess[eta]) + curvatura(egrad,eta).
function [Hrh, store] = hess(XX, eta, store)
    if nargin < 3 || ~isfield(store, 'Eamb')
        store.Eamb = egrad_sparse(XX);        % egrad sparso, una volta per punto
    end
    Heh = ehess_sparse(XX, eta);              % P(eta) sparso
    Hrh = problem.M.ehess2rhess(XX, store.Eamb, Heh, eta);
end

%% additional

% Prodotto di Khatri-Rao
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

% Prodotto interno sullo spazio tangente: identico a
% M.inner del factory, ma calcolato nella forma stabile <V*C_i, W*C_i>
%
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

% Trasporto vettoriale da X a Y: stessa matematica di M.transp del factory ma senza mai costruire il
% tensore ambiente pieno.
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


%NUOVO FINO A QUI ----------------------------

    % Define the Euclidean gradient of the cost function, that is, the
    % gradient of f(X) seen as a function of X without rank restrictions.
%     problem.egrad =  @egrad;
%     function [g, store] = egrad(X, store)
%         if ~isfield(store, 'PXmPA')
%             Xfull = full(X.X);
%             store.PXmPA = P.*Xfull - PA;
%         end
%         g = store.PXmPA;
%     end
    
    % Define the Euclidean Hessian of the cost at X along a vector eta.
%     problem.ehess = @ehess;
%     function H = ehess(X, eta)
%         ambient_H = problem.M.tangent2ambient(X, eta);
%         H = P.*ambient_H;
%     end

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

    % --- Check gradient e hessian in X0
    % 
    if isfield(options, 'check_derivatives') && options.check_derivatives
        fprintf('--- checkgradient (slope atteso ~2) ---\n');
        checkgradient(problem, X0);
        fprintf('--- checkhessian (slope atteso ~3, simmetria ~1e-12) ---\n');
        checkhessian(problem, X0);
        drawnow;
    end

    % --- Modalita' "solo gradiente": no ottimizzazione , for debug
    if isfield(options, 'grad_only') && options.grad_only
        Xtr = X0;
    else
        % --- Scelta del solver: default trust-regions (usa l'Hessiano) ---
        % or options.solver = 'conjugategradient' and 'steepestdescent'.
        if isfield(options, 'solver') && ~isempty(options.solver)
            solver = lower(options.solver);
        else
            solver = 'trustregions';
        end
        switch solver
            case 'trustregions'
                Xtr = trustregions(problem, X0, options);
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
    %
    X0_out      = X0;
    rg0_out     = rg_init;
    norm_rg0_out = norm_rg0;

    % Assegnazione finale
    Xtr = Xtr.X;

    % Display some quality metrics for the computed solution
    %Xtrfull = full(Xtr.X);
    %Afull = tensor(A);
    %fprintf('||X-A||_F / ||A||_F = %g\n', norm(Xtrfull - Afull)/norm(Afull));
    %fprintf('||PX-PA||_F / ||PA||_F = %g\n', norm(P.*Xtrfull - PA)/norm(PA));

    %keyboard
    %Res = norm(P.*Xtrfull - PA)/norm(PA); %Relative residual on the mask
    Res = 0.0;
end
