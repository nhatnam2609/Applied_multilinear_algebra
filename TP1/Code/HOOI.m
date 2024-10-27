function [ G_final, A ] = HOOI( X, R ,max_iter, tol)
% Higher Order Orthogonal Iteration for Tensor Decomposition
% X       - input tensor
% R       - vector of target ranks for each mode
% max_iter- maximum number of iterations
% tol     - convergence tolerance


dims = size(X);
N = ndims(X);
A = cell(1, N);
G = X;
[~,A] = HOSVD( X, R ); %initialize A using HOSVD

AT = cellfun(@(x) x', A, 'UniformOutput', false); % Transpose all A in previous step
for iter = 1:max_iter % repeat until fit ceases to improve or maximum iterations exhausted
    for n = 1:N  % repeat for all An do
        AT_except_n = AT;
        AT_except_n{n} = eye(size(A{n}, 1)); % Replace n-th matrix with identity matrix to keep the size
%         AT_except_n = AT([1:n-1, n+1:end]);
        Y = tprod(X, AT_except_n);  % Project tensor X onto the subspace defined by all A except A{n}
        Yn = ndim_unfold(Y,n);  % Unfold Y with the n-th mode
        [U, ~] = svdtrunc(Yn); %get left singular vectors of Y(n)
        A{n} = U(:, 1:R(n)); % get Rn leading of left singular vectors
        
        AT{n} = A{n}'; % Update transpose matrix list
    end

    % Reconstruct the core tensor x using the updated A
    G_new = tprod(X, AT);
    X_reconstructed = tprod(G_new, A);
    Difference_norm = norm(X(:) - X_reconstructed(:)) ;
    Relative_norm = Difference_norm/norm(X(:));
    fprintf('Iteration %d: Relative_norm = %.4f\n', iter,Relative_norm );

    if Relative_norm < tol
        fprintf('Convergence achieved within tolerance at iteration %d.\n', iter);
        break;
    elseif iter > 1 && abs(prev_norm - Relative_norm) < 1e-6  % Check if improvement ceases
        fprintf('No significant improvement, stopping at iteration %d.\n', iter);
        break;
    end
    prev_norm = Relative_norm;  % Update previous norm for next iteration
end
G_final = G_new;
end

