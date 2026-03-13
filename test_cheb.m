% 

f = @(x,y) log(3 + x + y);

n = 7;
k = 5;
x = chebpts(n);
y = chebpts(n);
[xx, yy] = meshgrid(x, y);
fxy = f(xx, yy);

% Transform into che Chebyshev basis
C = zeros(n, n);
Cx = chebvander(x);
Cy = chebvander(y);
C = Cy \ fxy / Cx';

[U, S, V] = svd(C);
U = U(:, 1:k);
S = S(1:k, 1:k);
V = V(:, 1:k);

% Increase n, and extend
n = 2*n + 1;
U(n, 1) = 0; V(n, 1) = 0;

% Debug
%C = U*V';
%p = chebfun2(C, 'coeffs');

x = chebpts(n); y = chebpts(n);
Cx = chebvander(x); Cy = chebvander(y);
UV = Cy * U;
VV = Cx * V;
[xx, yy] = meshgrid(x, y);
fxy = f(xx, yy);