n = 32;
r = 4^3;

Core = tensor(randn(4,4,4));
for i = 1 : 3; Core = ttm(Core, diag(.5 .^ (1 : 4)), i); end
T = ttensor(Core, {randn(n,4), randn(n,4), randn(n,4)});
% A = rand(n,n,n);
A = full(T); A = A.data;

% r > 1?
I1 = randperm(n, r); I2 = randperm(n, r); I3 = randperm(n, r);

U1 = zeros(n, 0); U2 = zeros(n, 0); U3 = zeros(n, 0); C = [];
for i = 1 : r
    i1 = I1(i); i2 = I2(i); i3 = I3(i);
    
    % index 1
    u1 = A(:,i2,i3); u1 = u1(:);
    for j = 1 : i-1
        u1 = u1 - C(j) * U1(:,j) * U2(i2,j) * U3(i3,j);
    end
    u1 = u1 / u1(i1);

    % index 1
    u2 = A(i1,:,i3); u2 = u2(:);
    for j = 1 : i-1
        u2 = u2 - C(j) * U1(i1,j) * U2(:,j) * U3(i3,j);
    end
    u2 = u2 / u2(i2);

    % index 1
    u3 = A(i1,i2,:); u3 = u3(:);
    for j = 1 : i-1
        u3 = u3 - C(j) * U1(i1,j) * U2(i2,j) * U3(:,j);
    end
    C(i) = u3(i3);    
    u3 = u3 / u3(i3);

    U1 = [U1, u1]; U2 = [U2, u2]; U3 = [U3, u3];

    Core = zeros(i,i,i); for j = 1 : i; Core(j,j,j) = C(j); end
    T = ttensor(tensor(Core, [j j j]), {U1, U2, U3});
    err = norm(A - full(T)) / norm(A, 'fro');
    fprintf('rank = %d, err = %e\n', i, err);
end
