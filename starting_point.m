function Xt = starting_point(A,r)
%Generalization for d dimensional Tucker
tensor_dim = size(A);

    d = length(tensor_dim);
    %r dato
    
    I = cell(1,d);
    for k = 1 : d
        I{k} = randperm(tensor_dim(k), r);
    end
    
    C = [];
    
    U = cell(1,d);
    for k = 1 : d
        U{k} = zeros(tensor_dim(k), 0);
    end
    
    idx = repmat({':'},1,d);
    
    for i = 1 : r
    
    id_i = repmat({i},1,d);
    
        for k = 1 : d
            idx{k} = I{k}(i);
        end
            %i1 = I1(i); i2 = I2(i); i3 = I3(i);
            
        for q = 1 : d
            idx{q} = ':';
            
    
            u = A(idx{:});
            u = u(:);
    
            for j = 1 : i-1
    
                UU = U{q}(:,j);
                for h = 1:d
                    if h~=q
                        UU = UU*U{h}(idx{h},j);
                    end
                end
    
                u = u - C(j)*UU;
        
            end
    
            idx{q} = I{q}(i);
            
            if q == d
                C(i) = u(idx{q});
                %keyboard
            end
           
            u = u /u(idx{q});
            U{q} = [U{q}, u];
        end  
    
            Core = zeros(id_i{:});
            for j = 1 : i
                %keyboard
    
                id_j = repmat({j},1,d);
                Core(id_j{:}) = C(j); 
            end
       Xt = ttensor(tensor(Core, cell2mat(id_j)), U);    
       err = norm(A - full(Xt)) / norm(A, 'fro');
       fprintf('rank = %d, err = %e\n', i, err);
    end
