function G = tprod(X, A)
%TPROD Tensor product of a multidimensional array and a set of transformation matrices
%	G = TPROD(X, A)
%	
%	X  - tensor (multidimensional array)
%	A  - cell array containing matrices for each dimension of X
%	
%	G  - result of the product, effectively the core tensor G when A{n} contains
%	     the left singular vectors for each mode of the tensor X
%
G = X;  % Initialize the resulting tensor G to be the input tensor X
sizeX = size(X);
for i = 1:length(A)
	if ~isempty(A{i})
		sizeX(i) = size(A{i},1); % Update the size of the n-th dimension as per the transformation matrix A{n}
		Xn = ndim_unfold(G, i); % Unfold tensor X along the n-th dimension
		G = ndim_fold(A{i}*Xn, i, sizeX); % Multiply the unfolded tensor by matrix A{n} and fold it back
	end
end
