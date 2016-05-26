# Variational Gaussian mixture model for MATLAB (vbGMM)

This toolbox implements variational inference for Gaussian mixture models (vbGMM) as per Chapter 10 of *Pattern Recognition and Machine
 Learning* by C. M. Bishop (2006). Part of the code is based on a barebone [MATLAB  implementation](http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model) by Mo Chen. vbGMM contains a number of additional features:
 
 - Generate samples from the trained mixture model (*vbgmmrnd.m*)
 - Expected pdf of the trained vbGMM at any given point (*vbgmmpred.m*)
 - Support for bounded variables (data are transformed for inference to an unbounded space via a nonlinear transformation) -- to be implemented.
 - Generate marginal and conditional vbGMMs -- to be implemented.
 
The toolbox is still work in progress and currently incomplete.
