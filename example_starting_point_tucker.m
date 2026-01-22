% Idea: for a d-dimensional tensor Tucker approximation.
%
% We can:
%  for each mode j = 1, ..., d:
%   - choose r random fibers long mode j;
%   - orthogonalize them and use them as starting Tucker bases.

n = [8, 8, 8]; d = 3;
T = tenrand(n(1), n(2), n(3)).data;
r = 3;

for j = 1 : 3
    % Select r random indices in 1, ..., n^(d-1)
    I = randperm(numel(T) / n(j), r);

    T1 = permute(T, [j, 1:j-1, j+1:d]);
    T1 = reshape(T, n(j), numel(T1) / n(j));

    [U{j}, ~] = qr(T1(:, I), 0);
end