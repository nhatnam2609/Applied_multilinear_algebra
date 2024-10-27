function [A, lambda_r] = normalize_columns(A)
    % Normalize the columns of matrix A and store the norms in lambda
    % A: factor matrix to normalize
    % lambda_r: vector to store column norms
    
    [m, n] = size(A); 
    lambda_r = zeros(1, n); % Initialize the lambda vector for storing norms
    
    % Loop over each column and compute the 2-norm using the built-in norm function
    for i = 1:n
        lambda_r(i) = norm(A(:, i), 2);  % Compute the 2-norm of the i-th column
        A(:, i) = A(:, i) / lambda_r(i); % Normalize the i-th column
    end
end