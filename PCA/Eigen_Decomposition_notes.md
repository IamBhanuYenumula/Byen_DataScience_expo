Prerequisites to understand Eigen Decomposition: Some important matrix before jumping into Decompositions.
## Diagonal Matrix:
It is a type of square matrix where the entries outside the main diagonal are all zero; the main diagonal is from the top left to the bottom right of the square matrix.
1.Powers: The nth power of a diagonal matrix ( where n is a non-negative integer) can be obtained by raising each diagonal element to the power of n

![image](https://github.com/user-attachments/assets/d3bafd90-6f7e-4e67-8ce8-e5818442658d)

2.Eigenvalues: The eigenvalues of a diagonal matrix are just the values on the diagonal. The corresponding eigenvectors are the standard basis vectors.

![image](https://github.com/user-attachments/assets/4ece6e3f-71e1-4dbd-8d19-088b710e088c)

3.Multiplication by a Vector: When a diagonal matrix multiplies a vector, it scales each component of the vector by the corresponding elements on the diagonal.

![image](https://github.com/user-attachments/assets/1ee6bd6b-86dc-4cb7-b627-aa07b4afd9b8)

![image](https://github.com/user-attachments/assets/9e354f60-ce29-47e5-90dd-6da502c4de8b)

4.Matrix Multiplication: The product of two diagonal matrices is just the diagonal matrix with the corresponding elements on the diagonals multiplied.

![image](https://github.com/user-attachments/assets/843f024c-0f05-4beb-9228-5c6d71fd74b6)


## Orthogonal Matrix:

![image](https://github.com/user-attachments/assets/1ae81065-418e-4453-bba2-2b20b77e001b)

An orthogonal matrix is a square matrix whose columns and rows are orthogonal unit vectors (I;e orthonormal vectors), meaning that they are all of unit length and are at right angles to each other.
Perfect rotation, no scaling or shearing.

![image](https://github.com/user-attachments/assets/0ee9f875-03ed-49e3-b66b-98d8c84bee46)

Inverse Equals Transpose: The transpose of an orthogonal matrix equals its inverse, i.e., A^T = A^(-1). This property makes calculations with orthogonal matrices computationally efficient.

## Symmetric Matrix:
![image](https://github.com/user-attachments/assets/0db4c7da-28b6-4ffa-973b-ab09dc27bff2)

A symmetric matrix is a type of square matrix that is equal to its own transpose. In other words, if you swap its rows with columns, you get the same matrix.
	• Real Eigenvalues: The eigenvalues of a real symmetric matrix are always real, not complex
	• Orthogonal Eigenvectors: For a real symmetric matrix, the eigenvectors corresponding to different eigenvalues are always orthogonal to each other. If the eigenvalues are distinct, you can even choose an orthonormal basis of eigenvectors. 

![image](https://github.com/user-attachments/assets/6842ab8e-2349-4ded-9ec1-9b78d308955b)



## What is matrix composition ?
When we multiply matrices , what would be its geometric intuition ?
[ a b      [ e f         =  [ k  l
 c d ]      g  h ]            m  n ]

If we apply [ k l      transformation on any vector that is nothing but applying [ e f first and then applying  [ a b
		         m n ]                                                                g h ]                          c d ]
		
This operation is called matrix composition.


![image](https://github.com/user-attachments/assets/25f240d4-8dae-4c36-bb20-8d4631a80284)

## What is matrix decomposition ?
There are various ways of matrix decomposition, few of the famous decompositions are listed below.
 Eigen and SVD are majorly used in ML.
 
![image](https://github.com/user-attachments/assets/f5de3b9d-7cd6-4c55-a5c8-463b21d6df3c)

## Eigen Decomposition:

The eigen decomposition of a matrix A is given by the equation:
 A = VAV^-1

Where:
1.V is a matrix whose columns are the eigen vectors of A
2.A(lambda) is a diagonal matrix whose entries are the eigenvalues of A
3.V^-1 is the inverse of V

![image](https://github.com/user-attachments/assets/6702925e-3290-4696-b625-3e096cc8ec88)

![image](https://github.com/user-attachments/assets/3beecf28-0de8-4034-ae3e-c3760ceab0ca)

Assuming
1. Square matrix: Eigen decomposition is only defined for square matrices
2.Diagonalizability: For a n*n matrix it should have n linearly independent eigen vectors
In the above example we assumed A is a square matrix. What if A is a symmetric ?

## Eigen Decomposition of Symmetric Matrix:
If A is a symmetric matrix the decomposition formula still holds the same but it is called as spectral decomposition.

![image](https://github.com/user-attachments/assets/5e414a2f-1f70-4684-9571-c67a9cc7aec7)

![image](https://github.com/user-attachments/assets/d210ac07-fa53-4d9a-b12d-74cd2ca0ef66)

Video reference :Visualize Spectral Decomposition | SEE Matrix, Chapter 2
https://www.youtube.com/watch?v=mhy-ZKSARxI&list=PLWhu9osGd2dB9uMG5gKBARmk73oHUUQZS&index=4 

In a nutshell: For any symmetric matric the liner transformation will be: 3 components:
1. Rotate
2. Scale
3. Rotate
And this is what Eigen decomposition is.


In PCA we take Covariance matrix, which is a symmetric matrices and we do the same steps with the eigen values to get the unit vector of higher variance. Essentially behind the scenes its doing Eigen decomposition.
Let's break this down to see how these concepts interrelate in PCA:
1. Covariance Matrix:
In PCA, we start with a covariance matrix, which is indeed symmetric.
This matrix represents the pairwise covariances between the variables in our dataset.
2. Eigendecomposition:
PCA essentially performs eigendecomposition on this covariance matrix.
Eigendecomposition breaks down the matrix into eigenvectors and eigenvalues.
3. Eigenvectors and Eigenvalues:
The eigenvectors of the covariance matrix represent the directions of maximum variance in the data.
The corresponding eigenvalues represent the amount of variance explained by each eigenvector.
4. Principal Components:
The eigenvectors become the principal components.
They are ordered by their corresponding eigenvalues, from highest to lowest.
5. Variance Explained:
The eigenvalues directly relate to the amount of variance explained by each principal component.
The proportion of variance explained by each component is its eigenvalue divided by the sum of all eigenvalues.
6. Unit Vectors:
The eigenvectors are typically normalized to unit length, which is why we get unit vectors.
7. Dimensionality Reduction:
By selecting the top k eigenvectors (based on eigenvalues), we can reduce the dimensionality of the data while retaining the directions of maximum variance.


Key Points:
• The symmetry of the covariance matrix ensures that its eigenvectors are orthogonal, which is crucial for PCA.
• The eigendecomposition is the mathematical core of PCA, providing the directions (eigenvectors) and magnitudes (eigenvalues) of maximum variance.
• This process allows PCA to find the most important features or directions in the data, which often correspond to meaningful patterns or structures.
















 










