%Test to check the extension in d dimensions
close all; clear all; clc;
%EXAMPLE ----- 2 variables
% f = @(x,y) log(3 + 0.5*x + y);
% 
% n = 7;
% k = 5;
% x = chebpts(n);
% y = chebpts(n);
% [xx, yy] = meshgrid(x, y);
% fxy = f(xx, yy);
% 
% %keyboard
% % 
% F = tensor(fxy);
% 
% core_dims = [5,5];
% tensor_dims = [7, 7];
% 
% intervals = cell(1,2);
% intervals{1} = [-1,1];
% intervals{2} = [-1,1];
% 
% apprT = cheb_approx(core_dims, tensor_dims, F, intervals, f);

%%EXAMPLE ---- 3 variables
f = @(x,y,z) log(5 + 0.5*x + y + 0.3*z);

n = 7;
k = 5;
x = chebpts(n);
y = chebpts(n);
z = chebpts(n);
[xx, yy, zz] = ndgrid(x, y, z);
fxyz = f(xx, yy, zz);

keyboard

Frec = tensor(C);   % oppure Frec = C se C è già tensor
for j = 1:d
    Frec = ttm(Frec, Cx{j}, j);
end

err = norm(double(Frec) - double(Ftrue));
disp(err)

F = tensor(fxyz);


core_dims = [5, 5, 5];
tensor_dims = [7, 7, 7];

intervals = cell(1,3);
intervals{1} = [-1,1];
intervals{2} = [-1,1];
intervals{3} = [-1,1];

apprT1 = cheb_approx(core_dims, tensor_dims, F, intervals, f);
apprT2 = cheb_approx_fft(core_dims, tensor_dims, F, intervals, f);

%%EXAMPLE ---- 5 variables
% f = @(x1,x2,x3,x4,x5) log(7 + x1 + x1 + x3 + x4 + x5 );
% 
% total = 50;
% n = 20;
% k = 10;
% x1 = chebpts(n);
% x2 = chebpts(n);
% x3 = chebpts(n);
% x4 = chebpts(n);
% x5 = chebpts(n);
% [xx1, xx2, xx3, xx4, xx5] = ndgrid(x1, x2, x3, x4, x5);
% fxyz = f(xx1, xx2, xx3, xx4, xx5);
% 
% F = tensor(fxyz);
% 
% core_dims = k*ones(1,5);
% tensor_dims = total*ones(1,5);
% 
% intervals = cell(1,5);
% intervals{1} = [-1,1];
% intervals{2} = [-1,1];
% intervals{3} = [-1,1];
% intervals{4} = [-1,1];
% intervals{5} = [-1,1];
% 
% apprT = cheb_approx(core_dims, tensor_dims, F, intervals, f);
