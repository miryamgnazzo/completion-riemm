function [Res, Xtrfull] = tensor_completion(A,core_dims)
%function based on manopt low_rank_tensor_completion_embedded()

    if ~exist('tenrand', 'file')
        fprintf('Tensor Toolbox version 2.6 or higher is required.\n');
        return;
    end


    %A is a multidimensional tensor in MATLAB
    tensor_dims = size(A);
    %core dims contains the rank

    if length(tensor_dims)~=length(core_dims)
        fprintf('Missing rank information.\n');
        return;
    end

    % Random data generation with pseudo-random numbers from a 
    % uniform distribution on [0, 1].  
    %tensor_dims = [60 40 20];
    %core_dims = [8 6 5];
    total_entries = prod(tensor_dims);
    d = length(tensor_dims);


    % Generate a random mask P for observed entries:
    % P(i, j, k) = 1 if the entry (i, j, k) of A is observed,
    %            = 0 otherwise.
    fraction = 0.1; % Fraction of observed entries.
    nr = round(fraction * total_entries);
    ind = randperm(total_entries);
    ind = ind(1 : nr);
    P = false(tensor_dims);
    P(ind) = true;
    % Hence, we observe the nonzero entries in PA:
    P = tensor(P);
    Afull = tensor(A);
    PA = P.*A; 
    % Note that an efficient implementation would require evaluating A as a
    % sparse tensor only at the indices of P.

    
    
    % Pick the submanifold of tensors of size n1-by-...-by-nd of
    % multilinear rank (r1, ..., rd).
    r = max(core_dims);
    problem.M = fixedranktensorembeddedfactory(tensor_dims, r*ones(1,d));
    %Simplified version with equal ranks!
    
    % Define the problem cost function.
    % The store structure is used to reduce full tensor evaluations.
    % Again: proper handling of sparse tensors would dramatically reduce
    % the computation time for large tensors. This file only serves as a
    % simple starting point. See help for the Tensor Toolbox regarding
    % sparse tensors. Same comment for gradient and Hessian below.
    problem.cost = @cost;
    function [f, store] = cost(X, store)
        if ~isfield(store, 'PXmPA')
            Xfull = full(X.X);
            store.PXmPA = P.*Xfull - PA;
        end
        f = .5*norm(store.PXmPA)^2;
    end

    % Define the Euclidean gradient of the cost function, that is, the
    % gradient of f(X) seen as a function of X without rank restrictions.
    problem.egrad =  @egrad;
    function [g, store] = egrad(X, store)
        if ~isfield(store, 'PXmPA')
            Xfull = full(X.X);
            store.PXmPA = P.*Xfull - PA;
        end
        g = store.PXmPA;
    end
    
    % Define the Euclidean Hessian of the cost at X along a vector eta.
    problem.ehess = @ehess;
    function H = ehess(X, eta)
        ambient_H = problem.M.tangent2ambient(X, eta);
        H = P.*ambient_H;
    end

    %initial point -- general dimension d
    Xt = starting_point(A,max(core_dims));
    X0.X = Xt;
    Cpinv = cell(1, length(tensor_dims));
    for i = 1:length(tensor_dims)
       Cpinv{i} = pinv(double(tenmat(X0.X.core, i)));
    end
    X0.Cpinv = Cpinv;

    
%     n1 = tensor_dims(1);
%     n2 = tensor_dims(2);
%     n3 = tensor_dims(3);
%     
%     if any(core_dims > tensor_dims)
%         fprintf('Check estimated rank.\n');
%         return;
%     end
% 
%     % r > 1?
%     I1 = randperm(n1, r); I2 = randperm(n2, r); I3 = randperm(n3, r);
%     U1 = zeros(n1,0); U2 = zeros(n2,0); U3 = zeros(n3,0);
% 
%     for i = 1 : r
%         i1 = I1(i); i2 = I2(i); i3 = I3(i);
%         
%         % index 1
%         u1 = A(:,i2,i3); u1 = u1(:);
%         for j = 1 : i-1
%             u1 = u1 - C(j) * U1(:,j) * U2(i2,j) * U3(i3,j);
%         end
%         u1 = u1 / u1(i1);
%     
%         % index 2
%         u2 = A(i1,:,i3); u2 = u2(:);
%         for j = 1 : i-1
%             u2 = u2 - C(j) * U1(i1,j) * U2(:,j) * U3(i3,j);
%         end
%         u2 = u2 / u2(i2);
%     
%         % index 3
%         u3 = A(i1,i2,:); u3 = u3(:);
%         for j = 1 : i-1
%             u3 = u3 - C(j) * U1(i1,j) * U2(i2,j) * U3(:,j);
%         end
%         C(i) = u3(i3);    
%         u3 = u3 / u3(i3);
%     
%         U1 = [U1, u1]; U2 = [U2, u2]; U3 = [U3, u3];
%     
%         Core = zeros(i,i,i); for j = 1 : i; Core(j,j,j) = C(j); end
%         Xt = ttensor(tensor(Core, [j j j]), {U1, U2, U3});
%     end
% 
%     X0.X = Xt;
%     Cpinv = cell(1, length(tensor_dims));
%     for i = 1:length(tensor_dims)
%        Cpinv{i} = pinv(double(tenmat(X0.X.core, i)));
%     end
%     X0.Cpinv = Cpinv;

    

    % Options
    %X0 = problem.M.rand();
    options.maxiter = 3000;
    options.maxinner = 100;
    options.maxtime = inf;
    options.storedepth = 3;
    % Target gradient norm
    options.tolgradnorm = 1e-5; %*problem.M.norm(X0, getGradient(problem, X0));

    % Minimize the cost function using Riemannian trust-regions
    Xtr = trustregions(problem, X0, options);

    % Display some quality metrics for the computed solution
    Xtrfull = full(Xtr.X);
    %Afull = tensor(A);
    fprintf('||X-A||_F / ||A||_F = %g\n', norm(Xtrfull - Afull)/norm(Afull));
    fprintf('||PX-PA||_F / ||PA||_F = %g\n', norm(P.*Xtrfull - PA)/norm(PA));

    Res = [norm(Xtrfull - Afull)/norm(Afull), norm(P.*Xtrfull - PA)/norm(PA)];
    
end
