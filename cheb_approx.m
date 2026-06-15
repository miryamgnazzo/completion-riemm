function [apprT, xx] = cheb_approx(core_dims, tensor_dims, F, intervals, f)
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
Cx = cell(1,d);

%F = double(F);
n_points = size(F); %array of dimensions of the evaluation tensor F

for i = 1:d
    x = chebpts(n_points(i), intervals{i});
%   Cx{i} = chebvander(x);
    Cx{i} = chebvander_shifted(x,intervals{i});
end

% Transform into the Chebyshev basis
%C = tensor(F);
C = F;
for j = 1:d
   %keyboard
   % C = ttm(C, Cx{j} \ eye(size(Cx{j})), j);  % mode-j product %forse migliorabile
   C.U{j} = Cx{j} \ C.U{j};
   %[C.U{j}, R] = qr(C.U{j}, 0);
   %C.core = ttm(C.core, R, j);
end

for j = 1 : d
    C.U{j} = [ C.U{j} ; zeros(tensor_dims(j) - size(C.U{j}, 1), size(C.U{j}, 2)) ];
end

n = size(C);
for j = 1:d
   %keyboard
   % C = ttm(C, Cx{j} \ eye(size(Cx{j})), j);  % mode-j product %forse migliorabile
   xx = chebpts(n(j), intervals{j});
   Cx = chebvander_shifted(xx, intervals{j});
   C.U{j} = Cx * C.U{j};
   [C.U{j}, R] = qr(C.U{j}, 0);
   C.core = ttm(C.core, R, j);
end

apprT = C;
xx = false;

%fC = double(C);
%keyboard;

return;

% --- Tucker decomposition ---
%TT = tucker_als(C, core_dims); %

% err_cheb = norm(full(TT) - tensor(C)) / norm(tensor(C));
% fprintf('Relative error on C = %e\n', err_cheb);
% pause

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

%n = 2.*n_points + 1;
n = tensor_dims;

for i = 1:d
    W = UU{i};
    W(n(i),1) = 0;
    UU{i} = W;
end

%New evaluations in the cheb points
%Cx_new = cell(1,d);
%UV = cell(1,d);
% for k = 1:d
%     xx{k} = chebpts(n(k), intervals{k});
%     %Cx_new{k} = chebvander(x);
%     UV{k} = chebvander(xx{k})*UU{k};
% end
% %Evaluation of the function on a larger grid of points
% apprT = Core;
% for j = 1:d
%     apprT = ttm(apprT, UV{j}, j);   % mode-j product
% end


%saving xx for debug later
% xx = cell(1,d);
% in formato tensor, ma non tucker
% apprT = Core;
% for k = 1:d
%     xx{k} = chebpts(n(k), intervals{k});
%     apprT = ttm(apprT, chebvander(xx{k})*UU{k},k);   % mode-j product
% end

% in formato ttensor (tucker in tensor toolbox)
xx = cell(1,d);
UV = cell(1,d);
for k = 1:d
    xx{k} = chebpts(n(k), intervals{k});
    UV{k} = chebvander_shifted(xx{k}, intervals{k}) * UU{k};
    %UV{k} = chebvander(xx{k}) * UU{k};
end
apprT = ttensor(Core, UV);

% Debug
%keyboard
% Confronto con funzione vera
% XX_points = cell(1,d);
% [XX_points{:}] = ndgrid(xx{:});
% Ftrue = f(XX_points{:});
% err = norm(full(apprT) - Ftrue);
% disp(err)
