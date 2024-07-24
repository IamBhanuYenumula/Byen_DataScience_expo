## KNN

### K-Nearest Neighbors: 
We can use this algorithm for both classification and regression problems.

### KNN - Intuition:
1. For a classification dataset
2. We start with a value of K, let's assume we took the value of k as 3
3. We have a query point, for which we need to find the class
4. To do so, we calculate the Euclidean distance of every datapoint from the query point and sort them to compare the top k class values, based on the majority of the class values from the k values we predict the class value of the query point.
5. For a regression problem we take the average of the k values to predict the output variable

### How to select K ?
It depends on the dataset that we are using, but to find out what k value is best suited for a given dataset, we have 2 approaches:
1. Heuristic approach
2. Experimentation 

In Heuristic we take √n , where n = number of observations
Note: we could avoid taking even k value
In Experimentation, we use cross-validation and select the model which gives the best accuracy score.

### What is Decision Surface or Decision Boundary ?
A tool used to understand the performance of a classification algorithm.
We have a library called "mlxtend" , this is used to plot decision surfaces.

![image](https://github.com/user-attachments/assets/fd3edcb6-435d-4b60-ae9f-23625ad96a6b)

Decision Boundaries are based on Voronoi diagram , which is a partitioning of a plane into regions based on distance to points in a specific subset of the plane.

### Overfitting and Underfitting in KNN:
1. When k value is very small, that will result in overfitting
2. If the k value is very large that will result in underfitting

### Limitations of KNN:
1. Avoid using KNN's when you have large datasets (KNN = Lazy learning technique)
		a. Reason: The majority of the action in this algorithm is happening in prediction phase and the training phase is nothing but storing the values. So computationally, prediction phase takes more time. That is why it is called as Lazy learning technique.
2. Avoid in case of high dimensions in the dataset. (Curse of dimensionality )
3. Sensitive to Outliers
4. Non-homogeneous scales : Scaling that is not consistent across all dimensions of a dataset
5. Imbalanced Datasets: An imbalanced dataset is characterized by a disproportionate distribution of classes. For example, in a binary classification problem, if one class has 90% of the samples and the other has only 10%, the dataset is considered imbalanced
6. KNN fails if we use it for Inference and not for Prediction (acts like a Block box model)

What is KNN Regressor ?
It is a variant of KN, which is used for regression problems. We follow the same steps till we find out the k values and at the end we take out the average of the k values to predict the output variable.


### Hyperparameters in KNN:
	• n_neighbors : default = 5
	• Weights : { 'uniform', 'distance'} default = 'uniform'
		○ Uniform: All points in each neighborhood are weighted equally
		○ Distance: Weight points by the inverse of their distance. In this case, closer neighbors of a query point will have a greater influence than neighbors which are further away
		Note: When we set weights to distance, we are doing weighted KNN.
	• Metric: Minkowski ( we need p value ) default value of p is 2 
	• Algorithm: {'auto','ball_tree','kd_tree','brute'}, default = 'auto'
	
### What is " Weighted KNN " ?
It is a variation of KNN, in which we calculate weights for every datapoint with respect to query point to predict output variable. Formula for calculating weight = 1/distance.

What are the types of Distances?
1. Euclidean (p = 2)
2. L2 norm
3. Manhattan (p = 1)
   
In metric hyperparameter we use minkowski and p value, the default p value is 2 which means it is using "Euclidean distance" to calculate from query point to data points. If we change this p- value we could apply different type of distance to KNN algorithm.

"Euclidean distance": The shortest distance between 2 points is called Euclidean distance. Formula = √ (x2 -x1)^2 - (y2-y1)^2.

For n dimensions, to calculate Euclidean distance, the formula is :
Dist = (∑ I = 1 to n (x2i -x1i)^2 ) ^1/2
![image](https://github.com/user-attachments/assets/b5d3fe7c-068d-4c3a-af13-2b9f1aceb3d1)

This is called as L2 norm.

L2 norm : It is basically Euclidean distance from the origin.

### Manhattan distance: 
It's the sum of the absolute differences between the coordinates of two points
![image](https://github.com/user-attachments/assets/b134322d-57c5-46cc-ae44-5bc7b1c77ee5)

Formula: For two points (x1, y1) and (x2, y2) in a 2D plane, the Manhattan distance is |x1 - x2| + |y1 - y2|

### Why do we need Manhattan distance when we have Euclidean distance, are there any problems with Euclidean distance ?

1. Curse of dimensionality: In high-dimensional spaces, Euclidean distance becomes less meaningful. The relative difference between the nearest and farthest points converges to 0 as dimensionality increases, making it difficult to discriminate between points.
2. Lack of variation in high dimensions: For high-dimensional data, the ratio of distances between the nearest and farthest neighbors to a given point approaches 1, resulting in little variation between distances of different data points.
3. Double zero problem: In ecological data, Euclidean distance is sensitive to the "double zero problem" where species absent in both compared communities contribute to similarity, even though this may not be ecologically meaningful.
4. Not suitable for sparse data: Euclidean distance may not be appropriate for sparse high-dimensional data, which is common in many real-world applications.
5. Sensitivity to data transformations: The impact of the double zero problem becomes even stronger when species composition data are transformed (e.g., log transformation or presence-absence transformation).
6. Not asymmetric: Unlike some other distance measures, Euclidean distance treats double presence and double absence equally, which may not be desirable for certain types of data like species composition.
7. May not capture relevant patterns: In some cases, especially with high-dimensional or ecological data, Euclidean distance may fail to capture the most relevant patterns or similarities between data points.
	
### What is the time and space complexity of KNN ?
We know that KNN is called as  (KNN = Lazy learning technique) algorithm for prediction. 
For a given data the time and space complexity of KNN will be O(nd).
Where, 
 n -> number of rows in training data
 d - > number of features

That is why we have certain improvements under "algorithm" hyperparameter. For example, using KD-tree the time complexity is reduced to O(d, log n).









