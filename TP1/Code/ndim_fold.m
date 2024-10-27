function X = ndim_fold(Ai, n, sizeX)
    % NDIM_FOLD Restore a matrix into a multidimensional array with the given size
    %
    % X = NDIM_FOLD(Ai, n, sizeX)
    % Ai   - matrix to be folded
    % n    - dimension along which the tensor was unfolded
    % sizeX - size of the original tensor

    % Calculate total elements in the tensor
    num_elements = prod(sizeX);

    % Ensure the product of dimensions
    if prod(size(Ai)) ~= num_elements
        error('The product of dimensions does not match the number of elements in the original tensor.');
    end

    % Reshape Ai into the tensor of the correct size
    X = reshape(Ai, sizeX([n, 1:n-1, n+1:end]));

    % Permute to swap the dimensions back to their original order
    order = [2:n, 1, n+1:length(sizeX)];  
    X = permute(X, order);
end
