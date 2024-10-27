function Xn = ndim_unfold(X, mode)
    % Unfold tensor X along a specified mode (1, 2, or 3)
    switch mode
        case 1
            Xn = reshape(permute(X, [1 2 3]), size(X, 1), []);
        case 2
            Xn = reshape(permute(X, [2 1 3]), size(X, 2), []);
        case 3
            Xn = reshape(permute(X, [3 1 2]), size(X, 3), []);
        otherwise
            error('Invalid mode.');
    end
end