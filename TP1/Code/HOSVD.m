function [ G, A ] = HOSVD( X, R )
%HOSVD Higher Order Singular Value Decomposition
%   [G, A, sv, tol] = HOSVD(X, R, tol)
%   
%   X   - Input tensor, a multidimensional array
%   R   - Rank in each mode specifying the number of singular values 
%         and vectors to retain 
%
%   G   - Core tensor such that X approximates tprod(G, A)
%   A   - Cell array of matrices containing the left singular vectors 
%         for each dimension
%   sv  - Cell array containing the singular values for each mode
%
%   Example:
%   [G, A, sv] = HOSVD(rand(10,10,10), [3, 3, 3])
    M = size(X);
    P = length(M);  % Number of modes of the tensor X

    if nargin < 2
        R = ones(1,P);  % Default rank if not specified
    end
    
    A = cell(1,P);    % Cell array to store factor matrices A^{(n)}
    AT = cell(1,P);   % Cell array to store the transpose of A^{(n)}
    
    for i = 1:P
        if R(i)
            Xi = ndim_unfold(X, i);  % Unfold tensor X along mode i
            [U, ~] = svd(Xi, 'econ');  % Perform SVD on the unfolded tensor
            Ai = U(:, 1:R(i));  % Retain the first R(i) singular vectors
            A{i} = Ai;          % Store the singular vectors in A^{(n)}
            AT{i} = Ai';        % Store the transpose of A^{(n)}
        else
            A{i} = [];
            AT{i} = [];
        end
    end
    
    G = tprod(X, AT);  % Compute the core tensor G using tprod
end

