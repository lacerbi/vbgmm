function varargout = vbgmmrnd(vbmodel,n)
%VBGMMRND Draw samples from posterior variational Gaussian mixture model.

% Predict label and responsibility for Gaussian mixture model trained by VB.
% Input:
%   X: d x n data matrix
%   model: trained model structure outputed by the EM algirthm
% Output:
%   label: 1 x n cluster label
%   R: k x n responsibility
% Written by Mo Chen (sth4nth@gmail.com).
alpha = vbmodel.alpha; % Dirichlet
beta = vbmodel.beta;   % Gaussian
m = vbmodel.m;         % Gaussian
nu = vbmodel.nu;       % Wishart
U = vbmodel.U;         % Wishart 
logW = vbmodel.logW;
[d,k] = size(m);

nd = size(n);
n = prod(n);

p = alpha./sum(alpha);
r = mnrnd(n,p); % To be fixed, do not use Stats toolbox

X = zeros(d,n);

for i = 1:k
    if r(i) == 0; continue; end
    
    offset = sum(r(1:i-1));
    di = [];
    Sigma = zeros(d,d,r(i));
    % Sample lambda
    S = U(:,:,i)'*U(:,:,i);  
    for j = 1:r(i)
        if ~isempty(di)
            L = iwishrnd(S,nu(i),di);
        else
            [L,di] = iwishrnd(S,nu(i));
        end
        Sigma(:,:,j) = beta(i)*L/nu(i);
    end
     X(:,offset+(1:r(i))) = mvnrnd(m(:,i)',Sigma)';    
    
end

% Transformation of constrained variables
if isfield(vbmodel,'prior') && ~isempty(vbmodel.prior) ...
        && isfield(vbmodel.prior,'LB') && isfield(vbmodel.prior,'UB')
    LB = vbmodel.prior.LB;
    UB = vbmodel.prior.UB;
    X = vbtransform(X,LB,UB,'inv');
end

varargout{1} = X;

end