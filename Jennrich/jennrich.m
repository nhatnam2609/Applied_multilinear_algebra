function [A_hat, B_hat, C_hat,lambda] = jennrich(X, R)
    % Jennrich decomposition function
    % Input:
    %   X - input tensor of size (I, J, K)
    %   R - target rank
    % Output:
    %   A_hat, B_hat, C_hat - estimated factor matrices

    % Get dimensions of the tensor
    I = size(X, 1);
    J = size(X, 2);
    K = size(X, 3);

    % Generate random vectors x and y of size K
    x = rand(K, 1);
    y = rand(K, 1);

    % Initialize random matrices A, B, and C
    A = rand(I, R);
    B = rand(J, R);
    C = rand(K, R);

    % Initialize matrices Xx and Xy
    Xx = zeros(I, J);
    Xy = zeros(I, J);

    % Step 1: Contract the tensor along the third mode with x and y
    for i = 1:R
        Xx = Xx + (C(:, i)' * x) * (A(:, i) * B(:, i)');
        Xy = Xy + (C(:, i)' * y) * (A(:, i) * B(:, i)');
    end

    % Step 2: Eigendecomposition to find A_hat
    [Vx, Dx] = eig(Xx * Xx');
    A_hat = Vx;
    % Step 3: Eigendecomposition to find B_hat
    [Vy, Dy] = eig(Xy' * Xy);
    B_hat = Vy;

    % Step 4: Compute C_hat
    V3 = (B_hat' * B_hat) .* (A_hat' * A_hat);
    X3 = ndim_unfold(X, 3); % Unfold X along the third mode
    C_hat = X3 * khatrirao_prod(B_hat, A_hat) * pinv(V3);
    % Step 5: Calculate lambda as scaling factors
    lambda = zeros(1, R);
    for r = 1:R
        lambda(r) = norm(A_hat(:, r)) * norm(B_hat(:, r)) * norm(C_hat(:, r));
        
        % Normalize A_hat, B_hat, and C_hat
        A_hat(:, r) = A_hat(:, r) / norm(A_hat(:, r));
        B_hat(:, r) = B_hat(:, r) / norm(B_hat(:, r));
        C_hat(:, r) = C_hat(:, r) / norm(C_hat(:, r));
    end
end
