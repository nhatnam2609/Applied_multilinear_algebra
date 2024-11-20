function X_reconstructed = reconstruct_tensor(A, B, C, lambda)
    % Get the sizes of the dimensions
    I = size(A, 1);  % Number of rows in A
    J = size(B, 1);  % Number of rows in B
    K = size(C, 1);  % Number of rows in C
    R = length(lambda);  % Rank of the decomposition (length of lambda)
    
    % Initialize the reconstructed tensor
    X_reconstructed = zeros(I, J, K);
    
    % Loop over each slice (third dimension of the tensor)
    for r = 1:R
        % Accumulate contributions from each rank
        for k = 1:K
            % Compute outer product A(:, r) * B(:, r)' and scale by lambda(r) and C(k, r)
            X_reconstructed(:,:,k) = X_reconstructed(:,:,k) + lambda(r) * (A(:,r) * B(:,r)') * C(k,r);
        end
    end

end