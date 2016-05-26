function varargout = vbgmmpred(vbmodel,X,flag)
%VBGMMPRED Posterior prediction for variational Gaussian mixture model.

% Predict label and responsibility for Gaussian mixture model trained by VB.
% Input:
%   X: d x n data matrix
%   model: trained model structure outputed by the EM algirthm
% Output:
%   label: 1 x n cluster label
%   R: k x n responsibility

% Author:   Luigi Acerbi
% Email:    luigi.acerbi@gmail.com
%
% Partially based on code written by Mo Chen:
% http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model

if nargin < 3 || isempty(flag); flag = 0; end


alpha = vbmodel.alpha; % Dirichlet
beta = vbmodel.beta;   % Gaussian
m = vbmodel.m;         % Gaussian
nu = vbmodel.nu;       % Wishart
U = vbmodel.U;         % Wishart 
logW = vbmodel.logW;
n = size(X,2);
[d,k] = size(m);

% Compute predictions only for variables within bounds
LB = vbmodel.prior.LB;
UB = vbmodel.prior.UB;
idx = all(bsxfun(@gt, X, LB) & bsxfun(@lt, X, UB),1);

X = vbtransform(X(:,idx),LB,UB,'dir'); % Change of variables
ntilde = size(X,2);

if nargout == 1

    % Compute posterior predictive density
    X = X';
    df = vbmodel.nu - d + 1;
    y = zeros(ntilde,1);
    p = vbmodel.alpha/sum(vbmodel.alpha);
    for i = 1:k
        if df(i) == Inf     % Multivariate normal
            S = U(:,:,i)'*U(:,:,i);
            y = y + p(i)*mvnpdf(X, vbmodel.m(:,i)', S);
        elseif df(i) <= 0   % Multivariate uniform ellipsoid
            S = U(:,:,i)'*U(:,:,i);
            y = y + p(i)*mvuepdf(X, vbmodel.m(:,i)', S);
        else
            C = (U(:,:,i)'*U(:,:,i))*((beta(i)+1)/(beta(i)*df(i)));
            s = sqrt(diag(C));
            Xres = bsxfun(@rdivide, bsxfun(@minus, X, vbmodel.m(:,i)'), s');
            y = y + p(i)*mvtpdf(Xres, C, df(i))/prod(s);
        end
    end
    if ~flag        
        y = y./exp(sum(vbtransform(X',LB,UB,'lgrad'),1))';
    end

    varargout{1} = NaN(n,1);
    varargout{1}(idx) = y;
    
else
    
    % Compute posterior predictive responsibility (might need to transform?)
    EQ = zeros(ntilde,k);
    for i = 1:k
        Q = (U(:,:,i)'\bsxfun(@minus,X,m(:,i)));
        EQ(:,i) = d/beta(i)+nu(i)*dot(Q,Q,1);    % 10.64
    end
    ElogLambda = sum(psi(0,0.5*bsxfun(@minus,nu+1,(1:d)')),1)+d*log(2)+logW; % 10.65
    Elogpi = psi(0,alpha)-psi(0,sum(alpha)); % 10.66
    logRho = -0.5*bsxfun(@minus,EQ,ElogLambda-d*log(2*pi)); % 10.46
    logRho = bsxfun(@plus,logRho,Elogpi);   % 10.46
    logR = bsxfun(@minus,logRho,logsumexp(logRho,2)); % 10.49
    R = exp(logR);
    z = zeros(1,ntilde);
    [~,z(:)] = max(R,[],2);
    [~,~,z(:)] = unique(z);
    
    varargout{1} = NaN(1,n);
    varargout{2} = NaN(n,k);    
    varargout{1}(idx) = z;
    varargout{2}(idx,:) = R;
end