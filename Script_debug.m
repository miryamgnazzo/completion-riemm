%%Debug e prove di controllo
%%First example: up to 3 parameters + time
nQ = 5;
en = (0:nQ)';

pi0 = zeros(nQ+1,1);
pi0(end)   = 0.5;
pi0(end-1) = 0.5;

%Intervalli
intervals = cell(1,4);
intervals{1} = [0.5, 3.0];    % t
intervals{2} = [1.0, 1.5];    % tau
intervals{3} = [0.05, 0.15];  %gamma
intervals{4} = [0.1, 0.99]; %c

core_dims = 5*ones(1,4);
final_dims = 128*ones(1,4);
sizes = [8, 32, 64];


%Second example: up to 6 parameters + time
% nQ = 4;
% en = zeros(nQ+1,1);
% en(1) = 1; en(2) = 2;
% 
% pi0 = zeros(nQ+1,1);
% pi0(3) = 1;
% 
% % Intervalli
% % intervals = cell(1,5);
% intervals = cell(1,7);
% intervals{1} = [0.5, 3.0];    % t
% intervals{2} = [1.0, 1.5];    % tau
% intervals{3} = [0.05, 0.15];  %gamma
% intervals{4} = [0.1, 0.99];   %c
% intervals{5} = [0.01, 0.1];   %lambda
% intervals{6} = [0.1, 0.99];   %p1
% intervals{7} = [0.1, 0.99];   %p2
% 
% %Opzione con 5 parametri
% core_dims = 5*ones(1,5);
% final_dims = 40*ones(1,5); %con 64 non ce la fa
% sizes = [8, 16, 32];

%Opzione con 7 parametri
% core_dims = 3*ones(1,7);
% final_dims = 10*ones(1,7); %con 20 si blocca
% sizes = [5, 6, 8];

% n0 = 4;
% t0 = chebpts(n0, intervals{1});
% tau0   = chebpts(n0, intervals{2});
% gamma0 = chebpts(n0, intervals{3});
% 
% A = zeros(n0);
% 
% for i = 1 : length(t0)
%     for j = 1 : length(tau0)
%         for k = 1: length(gamma0)
%             G = evalQ(nQ, tau0(j),gamma0(k));
%             A(i,j,k) = en'*expmv(t0(i), G, pi0);
%         end
%     end
% end