function X = vbgmmrnd(vbmodel,n)
%VBGMMRND Draw samples from posterior variational Gaussian mixture model.
%
%   X = VBGMMRND(VBMODEL) returns a random vector drawn from the variational 
%   Gaussian mixture model VBMODEL.
%
%   X = VBGMMRND(VBMODEL,N) returns N random vectors in a D-by-N matrix.
%
%   See also VBGMMFIT, VBGMMPDF.

% Author:   Luigi Acerbi
% Email:    luigi.acerbi@gmail.com

if nargin < 2 || isempty(n); n = 1; end

alpha = vbmodel.alpha; % Dirichlet
beta = vbmodel.beta;   % Gaussian
m = vbmodel.m;         % Gaussian
nu = vbmodel.nu;       % Wishart
U = vbmodel.U;         % Wishart 
[d,k] = size(m);

% nd = size(n);
n = prod(n);

p = alpha./sum(alpha);
r = multinomrnd(n,p);

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
    if any(isfinite(LB)) || any(isfinite(UB))
        X = vbtransform(X,LB,UB,'inv');
    end
end

end

%--------------------------------------------------------------------------
function r = multinomrnd(n,p)
%MULTINOMRND Random vectors from the multinomial distribution.

P = [0,cumsum(p,2)];
m = rand(n,1);
r = histc(m,P);

end
