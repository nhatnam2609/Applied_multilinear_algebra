# Tensor Decomposition in MATLAB

## Project Description

This repository contains MATLAB implementations for tensor decomposition algorithms: CP-ALS, HOSVD, and HOOI. These algorithms are crucial for applications such as signal processing, computer vision, and data mining where multidimensional data needs to be analyzed and simplified.

## Algorithms Implemented

- **CP-ALS**: Canonical Polyadic Decomposition using Alternating Least Squares (ALS) focuses on breaking a tensor down into a sum of rank-one tensors.
- **HOSVD**: Higher-Order Singular Value Decomposition provides a robust method for decomposing a tensor into orthogonal matrices and a core tensor, which simplifies the tensor structure significantly.
- **HOOI**: Higher-Order Orthogonal Iteration is an iterative refinement of HOSVD that enhances the approximation accuracy by iteratively optimizing the core tensor and the component matrices.

## Usage

Each algorithm is implemented in its own MATLAB file with supporting functions to perform necessary operations such as the Khatri-Rao product, tensor unfolding and folding, and normalization. To use these scripts, follow these steps:

1. **Clone this repository**

2. **Navigate to the repository directory** and open MATLAB in this directory.

3. **Run the test scripts** to see examples of tensor decompositions:
- `test_CP_ALS.m` for CP-ALS
- `test_HOSVD.m` for HOSVD
- `test_HOOI.m` for HOOI

4. **Modify the test scripts** to apply the algorithms to your own data.

## File Descriptions

- `CP_ALS.m`: Implementation of the CP-ALS algorithm.
- `HOSVD.m`: Implementation of the HOSVD algorithm.
- `HOOI.m`: Implementation of the HOOI algorithm.
- `khatrirao_prod.m`: Function to compute the Khatri-Rao product.
- `ndim_fold.m` & `ndim_unfold.m`: Functions to fold and unfold tensors.
- `normalize_columns.m`: Normalizes columns of a matrix.
- `reconstruct_tensor.m`: Reconstructs a tensor from its decomposed matrices.
- `tprod.m`: Computes the tensor product for given matrices.
- `Introduction-to-Tensors.pdf`: A document providing a background on tensor theory.
- `Kolda-BaderTensor_decomposition.pdf`: Detailed explanation of tensor decomposition methods.

