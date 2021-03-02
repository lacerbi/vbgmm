function y = mvuepdf(X,Mu,Sigma)
%MVUEPDF Multivariate uniform ellipsoid probability density function (pdf).
%
% See here: http://www.astro.gla.ac.uk/~matthew/blog/?p=368

[n,d] = size(X);

X0 = bsxfun(@minus,X,Mu);

R = chol(Sigma);
logSqrtDetSigma = sum(log(diag(R)));
Z = exp(-0.5*d*log(pi) + gammaln(0.5*d+1) - logSqrtDetSigma);
xRinv = (X0 / R);
d2 = sum(xRinv.^2,2);

idx = d2 < 1;
y = zeros(n,1);
y(idx) = Z;
            
end
