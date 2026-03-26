function [apprT, xx] = cheb_approx_fft(core_dims, tensor_dims, F, intervals)
% Choose the starting point of the tensor completion using 
% chebychev expansions
 

% F contains the evaluation of the measure on a grid of Chebyshev points  
% f is the multivariate function we need to evaluate
% intervals is a cell array containng the intervals for the evaluation

d = length(core_dims);

if (nargin < 3)
    fprintf('evaluation to be implemented')
    return
%X_points = ngrid(x1,....,xd); %se dovesse serivere per meshgrid
end

%Cx contains the evaluation of the cheb points in the grid (maybe we can
%avoid it, via cosine transform)
%Cx = cell(1,d);

%n_points = size(F); %array of dimensions of the evaluation tensor F
% 
% for i = 1:d
%     x = chebpts(n_points(i), intervals{i});
%     Cx{i} = chebvander(x);
% end

%Coefficients in Chebyshev basis via fft
C = double(F);
for dim = 1:d
    C = chebvals2coeffs_dim_fft(C, dim);
end

C = tensor(C);

% --- Tucker decomposition ---
TT = tucker_als(C, core_dims); %Problema "Input matrix is badly conditioned".
Core = TT.core;
UU = TT.U;

% Alternativa a tucker_als di tensor toolbox
% UU = cell(1,d);
% for k = 1:d
%   M = double(tenmat(C,k)); % mode-k unfolding
%   [Ut,~,~] = svd(M);
%   UU{k} = Ut(:,1:core_dims(k)); %select first columns
% end
% 
% %Core tensor
% Core = C;
% for j= 1:d
%  Core = ttm(Core, UU{j}', j);
% end

% Approximation on a larger grid of points
n = tensor_dims;

for i = 1:d
    W = UU{i};
    W(n(i),1) = 0;
    UU{i} = W;
end

% in formato ttensor (tucker in tensor toolbox)
xx = cell(1,d);
UV = cell(1,d);
for k = 1:d
    xx{k} = chebpts(n(k), intervals{k});
    %UV{k} = chebvander(xx{k}) * UU{k};
    UV{k} = chebvander_shifted(xx{k}, intervals{k}) * UU{k};
end
apprT = ttensor(Core, UV);
