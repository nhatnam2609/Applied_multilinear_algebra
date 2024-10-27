X = rand(10,10,10);
[ G, A] = HOSVD(X, [3, 3, 3]);

% Ensure the tprod function and other related functions are correctly implemented
X_reconstructed = tprod(G, A);

% Calculate the norm of the difference between the original and the reconstructed tensor
difference_norm = norm(X(:) - X_reconstructed(:));
Relative_norm = difference_norm/norm(X(:));
% Display the difference_norm using fprintf
fprintf('Difference norm: %f\n', difference_norm);
fprintf('Relative norm: %f\n', Relative_norm);