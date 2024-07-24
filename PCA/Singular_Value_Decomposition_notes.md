Understanding Some special matrices as a prerequisite before jumping into understanding SVD.
## Non-square Matrix: (rectangular matrix)
For a square matrix, we know how the linear transformation will affect a vector. Any vector when it is multiplied with a square matrices it will linearly transform in the coordinate space. But, the question is :

![image](https://github.com/user-attachments/assets/925edd5a-30f7-4f2a-86e9-6fe632ad4a9a)

How will be the linear transformation of a non-square matrix ?
When we are dealing with non-square matrices, we will change/switch dimensions. 
A non-square matrix of 3 X 2. 
2 - > input space
3 - > output space

For example: for a 2 X 3 matrix: the input space is in 3D and it will send the output to 2D.

## Rectangular Diagonal Matrix:
A matrix that would be diagonal if it were square, but instead is rectangular due to extra rows or columns of zeros. 

![image](https://github.com/user-attachments/assets/a286e9c8-9870-4945-9b51-6907e29ace81)

This matrix will apply 2 transformation at a time (any order):
1. Scaling
2. Dimension switching

![image](https://github.com/user-attachments/assets/2ba6800f-63e3-495d-9c61-12a71196179b)

So, when we try to apply linear transformation on rectangular diagonal matrix, 
1.First, it will first change the dimension from 3D to 2D (like we see matrix 1 in the above image) 
2.Second, it will apply scale with the second matrix with a factor of a to x - axis and with a factor of b on y-axis.



![image](https://github.com/user-attachments/assets/a7b96cd1-bdc9-4186-81b4-496d709d09d2)

## What is SVD: (Singular value decomposition)

SVD is matrix decomposition/factorization method that decomposes a matrix into three other matrices. Given a matrix A, the singular value decomposition of A is usually written as:

A=U∑V^T

Here: 
	• U and V are orthogonal metrices. U is the left singular vectors and V is the right singular vectors.
	• ∑ is a diagonal matrix containing what we call the singular values.

### Applications of SVD:
	1. Machine Learning and Data Science : SVD is used in PCA, It is also used in various recommendation systems
	2. Natural Language Processing (NLP): SVD used in Latent Semantic Analysis(LSA)
	3. Computer Vision: SVD used in Image compression
	4. Signal Processing: Separate useful signals from noise
	5. Numerical Linear Algebra: SVD is used for matrix inversion and solving systems of linear equations.
	6. Psychometrics: Used in construction and scoring of psychological and educational tests
	7. Bioinformatics: used to analyze gene expression data
	8. Quantum Computing: Used in quantum state tomography to understand the state of a quantum system 

### SVD The Equation:
Given a matrix A, it can be decomposed into three matrices [ U, ∑, V^T ]
	• A can be any matrix but it should not be infinite.
	• U is Orthogonal
	• ∑ is a Rectangular Diagonal Matrix
	• V is again be Orthogonal

Regarding the shape of the matrices:

	• A = m x n
	• U = m x m
	• ∑ = m x n
V^T = n x n

![image](https://github.com/user-attachments/assets/4fe42093-2958-4092-aaeb-0744111c86c0)


But the question is: 
What is U ?, 
What is ∑ ? 
what is V ?

The answer for these questions is hidden in Eigen's Decomposition.

## Relationship with Eigen Decomposition:

NOTE: Any non square matrix can be transformed into square matrix by multiplying the matrix with its transpose matrix. By doing so the resultant matrix is not only a square matrix but also a symmetric matrix.
Example:

![image](https://github.com/user-attachments/assets/215a32dd-e3b5-4fc5-afa1-a89bed7fcec6)

![image](https://github.com/user-attachments/assets/b437a1af-451e-47ca-b323-dab995eba1ec)

![image](https://github.com/user-attachments/assets/6721174b-9a13-49ec-bf1a-51bb6b8a7376)


Therefore, U is a matrix whose columns contain eigen vectors of AA^T
V is a matrix whose columns contain eigen vectors of A^TA

A -> U and A -> V has indirect relationship with is connected through A.AT and AT.A and that is why we call U and V as singular vector with respect to A. Where: 
U is left singular matrix
V is right singular matrix

Now, we know what is U and V. Now the question is:

What is ∑ ?

![image](https://github.com/user-attachments/assets/2c99523e-4547-4498-ab32-e4edb2e5ee5b)

![image](https://github.com/user-attachments/assets/67fd5499-d9fc-49c5-9dd4-3c014fee0da8)


### Geometric Intuition:

For any matrix A (m x n) we can divide the transformation into 4 parts:
1. Counter clock wise rotation in input space
2. Changing Dimension or switching Dimension 
3. Applying scaling/stretching 
4. Rotating back in clock wise direction

How to calculate SVD in numpy?
u,s,v_t =  np.linalg.svd(A)

Where u,s,v_t are variables to store the returned values of this function.













