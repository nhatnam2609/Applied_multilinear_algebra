function KR = khatrirao_prod(A, B)
    % Computes the Khatri-Rao product of matrices A and B
    % A is of size I x R, B is of size J x R same number of columns
    [I, R1] = size(A);
    [J, R2] = size(B);
    if R1 ~= R2
        error('Matrices must have the same number of columns.');
    end
    KR = zeros(I * J, R1);
    for r = 1:R1
        KR(:, r) = kron(A(:, r), B(:, r));
    end
end
