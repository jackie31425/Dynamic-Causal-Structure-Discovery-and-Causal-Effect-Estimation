clear all,clc,close all
rng(100)
% addpath(genpath(pwd))
X_raw = csvread("cos_30_100.csv",0,0)
m=5
T=100
total_replicate=30
B_estimate = zeros(m,m,T,total_replicate)

for replicate = 1:total_replicate
    X=X_raw(:,(T*(replicate-1)+1):(T*(replicate)))
    
    % only consider the change of causal coefficients
    % Xt = Bt*Xt + Et, Et~N(0,R);
    % b_{i,j,t} = a_{i,j}*b_{i,j,t-1} + epilson_{i,j,t}, epilson_t~N(0,q_{i,j});
    
    

    
    
    %m = size(G0,1);         % number of variables
    m=5
    N1 =20;                % Number of particles used in CPF-SAEM
    numIter = 1000;          % Number of iterations in EM algorithms 
    % one may reduce the number of iterations to speed up the process
    kappa = 1;              % Constant used to compute SA step length (see below)
    
    % SA step length
    gamma = zeros(1,numIter);
    gamma(1:2) = 1;
    gamma(3:3499) = 0.96;
    gamma(3500:end) = 0.96*(((0:numIter-3500)+kappa)/kappa).^(-0.4);
    
    % if you have prior knowledge of the causal graph, you may modify B_Mask
    % according to your prior knowledge. B_Mask(i,j) = 0 means the edge from i
    % to j is fixed to zero.
    B_Mask = ones(m,m);
    B_Mask = B_Mask - eye(m);

    
    % Initialization of the parameters
    A_init = zeros(m,m);
    q_init = zeros(m,m);
    R_init = diag(0.2*ones(m,1));
    
    % initialize the parameters by Kalman filter
    [A_init,q_init,R_init] = KF_initialization(X);
    
    
    % Run the algorithms
    fprintf('Running CPF-SAEM (N=%i). Progress: ',N1); tic;
    [R,q,A,B] = cpf_saem1_new(numIter, X, N1, gamma,R_init,q_init,A_init,B_Mask);
    timeelapsed = toc;
    fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed);
    % estimated parameters (use the parameters derived in the last iteration)
    R_hat = R(:,:,end);
    q_hat = q(:,:,end);
    A_hat = A(:,:,end);
    B_hat = B;
    G_hat = ones(m,m); % the estimated causal graph
    for i = 1:m
        for j = 1:m
            mu = mean(B_hat(i,j,:));
            va = var(B_hat(i,j,:));
            if(abs(mu)<0.05 & abs(va)<0.05)
                G_hat(i,j) = 0;
            end
        end
    end
    G_hat = G_hat';
    B_estimate(:,:,:,replicate)=B_hat
end
%save('example1', 'X','B0','q0','A0','R0','G_hat','B_hat', 'q_hat', 'A_hat', 'R_hat');
csvwrite('matresult_30_100_nocons.csv',B_estimate);

