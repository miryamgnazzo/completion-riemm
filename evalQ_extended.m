%% Infinitesimal Generator Matrix
function Q = evalQ_extended(nreplicas, lambda, c1, c2, mu1, mu2)
% the working states are "nreplicas", "nreplicas-1", ..., 1,
% the last minus one is a standby state (where the system has
% the last change to recover),
% and the last state (absorbing) is system failure state


mu1 = 0.5;
mu2 = 0.5;

R = sparse(nreplicas+2,nreplicas+2);
for i = nreplicas : -1 : 2
    % i counts the number of working replicas
    R(nreplicas-i+1, nreplicas-i+2) = i*lambda*c1;
    R(nreplicas-i+1,nreplicas+1) = i*lambda*(1-c1);
    if i<nreplicas
        R(nreplicas-i+1,nreplicas-i) = mu1;
    end
end
R(nreplicas, nreplicas+1) = lambda;
R(nreplicas, nreplicas-1) = mu1;

R(nreplicas+1,1) = c2*mu2;% rejuvenation
R(nreplicas+1,nreplicas+2) = (1-c2)*mu2;

Q = (R-diag(R*ones(nreplicas+2,1)));
end

%% Per il momento tutto commentato
% % function Q = evalQ_extended(nreplicas, ...
% %                              lambda, cf, ...
% %                              cr, cr1, ...
% %                              lambda2, c2, ...
% %                              mu1, mu2)
% % %Fixing a few parameters
% % lambda2 = 0;
% % c2 = 0;
% % 
% % cr1 = 1;
% % %cr = 0.9;
% % 
% % mu1 = 0.5;
% % mu2 = 0.5;
% % 
% % nstates = nreplicas + 2;
% % 
% % R = sparse(nstates,nstates);
% % 
% % for i = nreplicas : -1 : 2
% % 
% %     row = nreplicas - i + 1;
% %     R(row,row+1) = R(row,row+1) ...
% %                  + cf*i*lambda;
% % 
% %     if i > 2
% %         R(row,row+2) = R(row,row+2) ...
% %                      + c2*nchoosek(i,2)*lambda2;
% %     else
% %         % i = 2 : covered double failure goes to state 0
% %         R(row,nreplicas+1) = R(row,nreplicas+1) ...
% %                            + c2*lambda2;
% %     end
% % 
% %     R(row,nreplicas+1) = R(row,nreplicas+1) ...
% %                        + (1-cf)*i*lambda;
% % 
% %     R(row,nreplicas+1) = R(row,nreplicas+1) ...
% %                        + (1-c2)*nchoosek(i,2)*lambda2;
% % 
% %     if i < nreplicas
% %         R(row,row-1) = cr1*mu1;
% %     end
% % 
% % end
% % 
% % %% State 1
% % 
% % row = nreplicas;
% % 
% % % successful repair: 1 -> 2
% % R(row,row-1) = cr1*mu1;
% % 
% % % last component failure: 1 -> 0
% % R(row,nreplicas+1) = lambda;
% % 
% % %% State 0
% % 
% % R(nreplicas+1,1) = cr*mu2;
% % 
% % R(nreplicas+1,nreplicas+2) = (1-cr)*mu2;
% % 
% % %% Generator
% % 
% % Q = R - diag(sum(R,2));
% % Q = full(Q);
% % end