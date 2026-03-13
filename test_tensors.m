d = 3;
r = 10 * ones(1, d);
n = 100 * ones(1, d);

rtrue = 25 * ones(1, d);

C = tenrand(rtrue);
for j = 1 : d
    D = diag((.5).^(0 : rtrue(j)-1));
    C = ttm(C, D, j);
end

% Tucker factor
U = cell(1, d);
for j = 1 : d
    [U{j}, ~] = qr(randn(n(j), rtrue(j)), 0);
end

T = ttensor(C, U);
fT = full(T);

% Tucker approximation by column selection
V = cell(1, d);
for j = 1 : d
    % Random selection of mode-j fibers
    ind = randsample(prod(n) / n(j), 5 * r(j));
    W = zeros(n(j), length(ind));
    for i = 1 : size(W, 2)
        % Indice da prendere, compattato
        I = cell(1, d-1);
        [I{:}] = ind2sub(n([1:j-1,j+1:d]), ind(i));
        I = [ I(1:j-1), ':', I(j:d-1) ];
        W(:, i) = squeeze(fT(I{:}));
    end

    [V{j}, ~, ~] = qr(W, 0);
    V{j} = V{j}(:, 1 : r(j));
end

CV = tensor(fT);
for j = 1 : d
    CV = ttm(CV, V{j}', j);
end

apprT = ttensor(CV, V);