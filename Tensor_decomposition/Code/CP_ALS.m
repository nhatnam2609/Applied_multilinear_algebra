function [lambda, A,B,C] = CP_ALS(X, R, max_iter, tol)
    [I, J, K] = size(X);
    
    A = rand(I, R);
    B = rand(J, R);
    C = rand(K, R);
    lambda = ones(R, 1);
    
    N = ndims(X);
    prev_norm = inf;
    for iter = 1:max_iter
            V1 = (C' * C) .* (B' * B);
            X1= ndim_unfold(X,1); 
            A_hat = X1*khatrirao_prod(C,B)*pinv(V1);
            [A,lambda_A] = normalize_columns(A_hat);
    
            V2 = (C' * C) .* (A' * A);
            X2= ndim_unfold(X,2); 
            B_hat = X2*khatrirao_prod(C, A) * pinv(V2);
            [B,lambda_B] = normalize_columns(B_hat);
    
            V3 = (B' * B) .* (A' * A);
            X3= ndim_unfold(X,3); 
            C_hat = X3*khatrirao_prod(B, A) * pinv(V3);
            [C,lambda_C] = normalize_columns(C_hat);
            X_reconstructed = reconstruct_tensor(A, B, C, lambda);
            Difference_norm = norm(X(:) - X_reconstructed(:)) ;
            % Display norms
            Relative_norm = Difference_norm/norm(X(:));
            fprintf('Iteration %d: Relative_norm = %.4f\n', iter,Relative_norm );
        
            % Check for convergence: if improvement is below the tolerance
            if abs(prev_norm - Relative_norm) < tol
                fprintf('Convergence achieved within tolerance at iteration %d.\n', iter);
                break;
            end
        
            % Update prev_norm for the next iteration
            prev_norm = Relative_norm;
            lambda = lambda_A;
    end

