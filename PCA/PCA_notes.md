## Principal Component Analysis (PCA):
It is a statistical technique used for dimensionality reduction while preserving as much variability as possible in a dataset. It transforms the original set of correlated variables into a smaller set of uncorrelated variables called principal components.
PCA is unsupervised ML problem. It’s a feature extraction technique and its main goal is to reduce curse of Dimensionality. 
PCA tries to get a higher dimension dataset to a lower dimension without losing its essence. 

## Core Benefits of using PCA:
	1. Faster execution of Algorithms 
	2. Visualization  
   For example if we have 10 features we cannot visualize the data, 
   but using PCA we can reduce the dimension to 3D and visualize it

For instance:
Let's say we have a real-estate price dataset with 2 input and one output column.
![image](https://github.com/user-attachments/assets/55ac8cf1-37a3-4479-afb9-4745d2a3d27c)

In terms of feature selection perspective, if we want to choose one input variable out of the 2 given input variables, which one would you choose?
For any individual with real-estate domain knowledge will go with "no of rooms" column, because it has more weight on the price compared to the other column. 

But what if we come across some other dataset in other domains with n columns, for which we need to choose more relevant columns like we did in the above example.
We can use a mathematical trick to select features.
We select the feature that has higher variance (spread) on its respective axis. 
For example:
When we plot the data in 2D and try to project the spread on x and y axis and call the spread as d and d'. We can compare the spread of d and d', and we can see that d > d' . 
We can select x axis as our feature since it has higher spread compared to the other feature.

![image](https://github.com/user-attachments/assets/5856f025-2d77-4caf-86bf-329ebeb51dce)

Let's say we have some changes in the data and we now have a new feature "No of wash rooms" in place of grocery shops. Now we can notice linear relation in data and we can see that d is comparable to d' (d = d' ).
Now we can't decide the feature to select based on the variance. This is a drawback of the feature selection. In cases like this we use feature extraction.

![image](https://github.com/user-attachments/assets/761d0011-af00-495a-87cd-d66091bfd40e)

If we look at the example dataset of real-estate in feature extraction POV, we can combine rooms and washrooms into a single variable called size of the room. But with our real world dataset with more number of features, we use PCA.
In Feature extraction,  PCA will try to find a new set of coordinate by rotating the existing space. By shifting the axis we can see that there is a change in the variance on new coordinate system. And we will select the feature with high variance on this new coordinate system. The axes on the new coordinate system are called "Principal Components". 
In the below image we see the variance of PCA 1 is high so we choose that variable from PCA 1 and transform the data.

![image](https://github.com/user-attachments/assets/a8efa835-04c8-4838-abb1-b4b9cecabf49)
Note: The number of principal components will always be <= to features.

### But why is variance/spread Important ?
Variance is crucial in Principal Component Analysis (PCA) because it helps to identify the directions in which the data varies the most. 
1.Capturing Maximizing Variability: 
PCA aims to transform the data into a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate (the first principal component), the second greatest variance on the second coordinate, and so on. This ensures that the principal components capture the most significant patterns in the data
2.Explained variance: 
The proportion of the total variance explained by each principal component indicates how much of the data's variability is captured by that component. 
Reducing Dimensions While Preserving Information: 
By focusing on the components with the highest variance, PCA reduces the number of dimensions needed to describe the data without losing significant information. This is particularly useful in simplifying datasets and improving computational efficiency
3.Improving Interpretability: 
Components with higher variance are more informative and hence more useful for understanding the underlying structure of the data. This makes it easier to interpret the results of PCA and to visualize high-dimensional data in lower dimensions
4.Model Performance:
In machine learning, using principal components with high explained variance can improve model performance by focusing on the most informative features and reducing noise and overfitting

variance is maximized in PCA to ensure that the principal components capture the most significant and informative patterns in the data, facilitating effective dimensionality reduction, improved interpretability, and better model performance.

### Problem Formulation:
What will PCA solve mathematically ? What is the mathematical objective function that PCA is trying to solve ?

Let's say we have a dataset in 2D and we aim to reduce it to 1D.
Now, we have to find a single axes, on which I can project my data so that I can get the same results as I used to get in 2D.
![image](https://github.com/user-attachments/assets/c845b805-e24a-4346-88f8-7b50c8a8b50b)

That axes could be this axis (orange) that we see on the image.
Let see what will PCA solve in this situation, by taking a single point in the data. Let's say the point is X and it has its coordinates (x , y) and we can also say that X is a vector and it has 2 components one in x direction and other in y direction .
![image](https://github.com/user-attachments/assets/05b0d9e3-8c37-4db9-a69b-43b9252ade05)

Now, we need to project this vector X on to another vector. The vector onto which our vector X will be projected, we are not worried about its magnitude we only needs its direction. 
So, we just need a "unit vector" . Let's represent it as vector u.
![image](https://github.com/user-attachments/assets/1656540f-5338-494f-8d4e-3344ebbe11ed)
We need to project the point x on to this unit vector.
Projections: The scalar projection of b onto a is the length of the segment AB. The vector projection of b onto a is the vector with this length that begins at the point A points in the same direction (or opposite direction if the scalar projection is negative) as a.
![image](https://github.com/user-attachments/assets/7907ebb5-8497-470e-8b6b-b1835bea9795)
Thus, mathematically, the scalar projection of b onto a is |b|cos(theta) where theta is the angle between a and b.
![image](https://github.com/user-attachments/assets/d720845e-99c2-42bc-a648-587ea8898325)

So, in our case, the formula would be u.x/|u| .
Since |u| = 1, u . X = uT.x
The new vector that is formed after the projection is called as x'. This will be a scalar, representing the length. If we do it for every point then we get a scalar for every individual point, which is representing the length.
Now, what is the suggested unit vector ?
We choose that unit vector from which our variance is maximum. 

To find out the variance we use this formula: Variance = ∑ I = 1 to n (uTxi - uTx(mean))^2 / n

This is called mathematical objective function.
By solving this mathematical objective function PCA will us the direction in which we get the maximum variance.

## Covariance and Covariance Matrix:
Variance is always a metric of a single axis. It fails to explain the relationship between variables. That is where covariance is useful.
Covariance is a statistical measure that quantifies the joint variability between two random variables. It indicates how two variables change together and the direction of their linear relationship.
This is similar to correlation, but in covariance it is not restricted between -1 to 1.
### Covariance matrix:
A covariance matrix is a square matrix that represents the covariance between pairs of variables in a multivariate dataset. It provides a comprehensive view of how different variables in a dataset relate to each other.
Properties of covariance matrix:
	• Symmetry: The covariance matrix is always symmetric.
	• Positive semi-definite: It is always positive semi-definite.
	• Diagonal elements: These represent the variances of individual variables.
	• Off-diagonal elements: These represent covariances between pairs of variables

Example of 3D covariance matrix :

![image](https://github.com/user-attachments/assets/b2c0ce8f-34e3-473e-b30f-5017e52ee581)

### How to calculate covariance matrix :
Step 1: Mean center all the columns
Step 2: covariance matrix = XT.X / (n-1)
![image](https://github.com/user-attachments/assets/311dc020-4ce7-42e4-9fc5-aab44d5352c6)

### What is the importance of covariance matrix ?
1. Spread: It will let us know the spread on all the axis, because it has the variance of every axis
2. Orientation: It tells us the relationship between two axis (+ positively corelated or - negatively corelated)

### Eigen decomposition of Covariance matrix: 
### What happens if we extract Eigen values or Eigen vectors from Covariance matrix ?
The Largest eigenvector of the covariance matrix always points into the direction of the largest variance of the data, and the magnitude of this vector equals the corresponding eigenvalue. 

### What are Eigen Vectors ?
Eigenvectors are vectors whose direction remains unchanged under a linear transformation, and are scaled by a corresponding eigenvalue.
Properties
1.Direction: Eigenvectors maintain their direction after the transformation; they are only scaled by the eigenvalue.
2.Linearly Independent: Eigenvectors corresponding to different eigenvalues are linearly independent.
3.Scaling: Eigenvectors can be scaled by any non-zero scalar and still be valid eigenvectors for the corresponding eigenvalue

### What are Eigen Values ?
After transformation, how much a Eigen vector is scaled or shrinked is known as Eigen value.
Eigenvalues are a special set of scalars associated with linear transformations of vector spaces, particularly in the context of matrix equations. 
They are crucial in understanding the behavior of these transformations 

### Step by step solution to reduce dimensionality using Eigen Decomposition:
1. Mean centering
2. Find Covariance matrix
3. Find Eigen value and Eigen vector for covariance matrix
4. The Eigen vector with high Eigen value will be our first Principal component and the list follows on
If we use first principal component we can transform to 1D if we use more than one we can transform to that many dimensions.

### How to find the optimum number of PCA in any given dataset?
"pca.explained_variance_ "will give us values but we need to convert that to percentage to check the percentage of variance explained by that eigen value on the total variance of the dataset.

Formula: (eigen value /sum of total_no_of_eigenValues ) * 100

![image](https://github.com/user-attachments/assets/b93c9b16-8097-4ef7-b987-2d5e2c692d2c)

If we do cumulative sum of the eigen values, the sum should reach to 90%. All the values that contribute to 90% are the essentially good number of PCA to consider in our mode. 
Which means all these PCA are able to explain 90% variance in our dataset.

 "pca.explained_variance_ration_" will give us the percentage ratio














