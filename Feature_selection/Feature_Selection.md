### What is a Feature ?
Feature is an input column.
In ML, we predict target column or output column by using Features

### What is feature selection?
In ML, if we have a dataset with 'p' features, we don’t use all the 'p' features to build our model. Instead we use a subset 's' of features to train out ML model. Where 'p'  >  's'.
For example, if we have 10 features in our dataset, we extract top 5 features to predict our target column. This process of Identifying those top contributing features out of all features and selecting them to use in ML model is called feature selection.

### What is the need of doing feature selection?
1. Curse of dimensionality :  It is a theory that says , ML model produces optimal results with certain number of features only, and if we try to use more features the model might not be a good fit and not perform effectively.
		a. Why this happens like this ? It happens because of a concept called "Sparsity".
2. Computation complexity : To reduce both space and time computation complexity
3. Interpretability: ML models are used for predictions and inference. To draw the inference if we have more features is a complex task, We end up considering all the factors to draw relationship. For example, studying 500 features is more complex when compared to studying 50 factors.

### What is Sparsity?
Sparsity refers to a situation where a significant portion of the data consists of empty, Zero or missing values. It occurs when only a small percentage of possible data points in a dataset contain actual, non-zero values. The majority of the data points are either zero, null or missing . For example, when we perform one hot encoding for any feature that results in a sparse matrix.

There are 2 types of sparsity:
1. Structural sparsity: Inherent in data structure itself, such as sparse matrices or graphs
2. Value sparsity: when most data values are zero or null, but can be potentially be filled or imputed

Measuring sparsity:
1.Sparsity ration: The proportion of zero or null values in the dataset
2.Density: The inverse of sparsity, representing the proportion of non-zero values


### Types of Feature Selection:
1. Filter based technique
2. Wrapper technique
3. Embedded techniques
4. Hybrid technique 

![image](https://github.com/user-attachments/assets/10bff9c2-c04c-482a-ba55-0d299174247d)


# Filter Based Feature Selection: 
Filter-based feature selection techniques are methods that use statistical measures to score each feature independently, and then select a subset of features based on these scores. These methods are called "filter" methods because they essentially filter out the features that do not meet some criterion.
Observations:
	• We focus on one feature at a time
	• We use statistical tests to filter out features
Statistical tests used in these technique:
1. Variance Threshold
2. Correlation
3. ANOVA
4. Chi-square
5. Mutual info

## Variance Threshold Method:
This techniques is applied for 2 types of features.
	1. Constant Feature
	2. Quasi Constant Feature

Constant Features: It is a variable that has the same value for all observations in a dataset. 
Quasi Constant Feature: A Quasi-constant feature is a variable that as the same value for a vast majority of observations, with only a few instances having different values.

#### Steps in Variance Threshold Method:
1. Define a threshold, e.g.: 0.1
2. Take out variance for all the features in the dataset
3. Check and compare the variance of each column with threshold value, if there are any columns with less variance compared to threshold then we can drop them

NOTE: When applying variance threshold, please make sure that all the columns are normalized (same scale). When data is normalized, setting a threshold value between 0.1 to 0.01 is a good strategy. 

#### Points to Consider:
1. Ignores Target Variable: Variance Threshold is a univariate method, meaning it evaluates each feature independently and doesn't consider the relationship between each feature and the target variable. This means it may keep irrelevant features that have a high variance but no relationship with the target, or discard potentially useful features that have a low variance but a strong relationship with the target
2. Ignores Feature Interactions: Variance Threshold doesn't account for interactions between features. A feature with a low variance may become very informative when combined with another feature
3. Sensitive to Data Scaling: Variance Threshold is sensitive to the scale of the data. If features are not on the same scale, the variance will naturally be higher for features with larger values. Therefore, it is important to standardize the features before applying Variance Threshold
4. Arbitrary Threshold Value: It's up to the user to define what constitutes a "low" variance. The threshold is not always easy to define and the optimal value can vary between datasets.

## Correlation:
Correlation-based feature selection is a method used to identify relevant features by measuring their correlation with the target variable and with other features. 
The main idea is to select features that are  highly correlated with the target variable but have low correlation with other features.
Pearson's correlation coefficient is commonly used for numerical variables. It measures the linear relationship between two continuous variables.
For  categorical variables or non-linear relationships, other correlation measures like "Spearman's rank correlation" or "mutual information" can be used.

In Correlation, we take 2 columns for example an independent column and a dependent column and we try to find out the linear relationship between those columns.
The result would be in between -1 to +1.
-1 = strong inverse linear relationship
+1 = strong positive linear relationship
If value is 0.9, it is said that there is a strong linear relationship between the variables and if the value is -0.9, that means there is a strong inverse linear relationship and if the value is zero, then it is said to have no relationship between those values.

### There are 2 ways of finding out the correlation between the variables. 
1.First approach: Let's assume we have input variables from f1,f2…fn and a target variable y. Then, we can take correlation of each input variable with respect to target variable y and decide a cutoff point to include or exclude that feature.
2.Second approach: Let's try to find out the correlation coefficients between all the columns. For example:( f1, f2), (f1, f3) …(f1, fn). By doing so, we get to know if there is multicollinearity in the dataset. 
For example: (f1-f2) = 0.9 value and (f1,f3) = 0.85, then we can just keep f1 column and exclude f2 and f3.

### Implementation:
1.  df.corr() function to form correlation matrix using pandas
2. Scikit-learn provides functions like" f_regression()" for Pearson's correlation and "mutual_info_regression() " for mutual information
3. "SelectKBest" and "SelectPercentile" can be used to select top features based on  correlation scores
	
### Steps involved:
1. Calculate correlation coefficients between each feature and the target variable
2. Calculating correlation coefficients between pairs of features
3. Selecting features with high correlation to the target and low correlation with other features
	
### Disadvantages:
1. Linearity Assumption: Correlation measures the linear relationship between two variables. If does not capture non-linear relationships well. If a relationship is nonlinear, the correlation coefficient can be misleading
2. Doesn't Capture Complex Relationships: Correlation only measures the relationships between two variables at a time. It may not capture complex relationships involving more than two variables.
3. Threshold Determination: Just like variance threshold, defining what level of correlation is considered "high" can be subjective and may vary depending on the specific problem or dataset.
4. Sensitive to Outliers: Correlation is sensitive to outliers.

## ANOVA:
It helps identify which features have a significant relationship with the target variable. It tests whether there are statistically significant differences between the means of two or more groups. In feature selection, we use it to determine if there is a significant relationship between each feature and the target variable.

### Implementation:
Scikit-learn provides the " f_classif " function for ANOVA F-value calculation, which can be used with "SelectKBest" or" SelectPercentile" for feature selection.

We use ANOVA when our input variables are numerical and output variable is categorical. Preferably if there are more than 2 categorical classes in output column. [one-way ANOVA]
(We can also use ANOVA with both input and output columns as numeric values)

In ANOVA, we take out each input column and study relationship with target column, to check if the relationship is strong or not, if it is not strong we exclude the variable.

![image](https://github.com/user-attachments/assets/fce80507-4d49-45ff-847e-f16d2852ab94)

### How to study the relationship?
By doing hypothesis testing, we need to take out the F-statistic or F-ratio and we then take out the p-value and compare the relationship of input and output variables.

#### Formula : MS between / MS within
To find out MS between and MS within we need to find out SS between and SS with in. 
For Degree of freedom: n is the number of rows and k is the number of categories

### Disadvantages:
1. Assumption of Normality: ANOVA assumes that the data for each group follow a normal distribution This assumption may not hold true for all datasets, especially those with skewed distributions.
2. Assumption of Homogeneity of Variance: ANOVA assumes that the variances of the different groups are equal. This is the assumption of homogeneity of variance( also known as homoscedasticity). If this assumption is violated, it may lead to incorrect results.
3. Independence of Observations: ANOVA assumes that the observations are independent of each other. This might not be the case in datasets where observations are related (e.g., time series data, nested data).
4. Effect of Outliers: ANOVA is sensitive to outliers. A single outlier can significantly affect the F-statistic leading to a potentially erroneous conclusion.
5. Doesn't Account for Interactions: Just like other univariate feature selection methods, ANOVA does not consider interactions between features. 

## Chi-Square:
Chi-square will analyses the relationship between input and output variables.  It is particularly useful when dealing with categorical features and classification problems. Chi-square works well with large datasets, as it can quickly evaluates the relationship between features and the target variable. It is used for initial screening of features in high-dimensional datasets.

### Implementation:
1. Form a contingency table (frequency table)
2. Calculate Chi-square statistic
3. Compute p-value
4. Compare the Chi-square statistic or p-value to rank the feature
5. Select the top k features based on their rankings


### Disadvantages:
1. Categorical Data Only: The chi-square test can only be used with categorical variables. It is not suitable for continuous variables unless they have been discretized into categories, which can lead to loss of information.
2. Independence of Observations: The chi-square test assumes that the observations are independent of each other. This might not be the case in datasets where observations are related(e.g., time series data, nested data)
3. Sufficient Sample Size: Chi-Square test requires a sufficiently large sample size. The results may not be reliable if the sample size is too small or if the frequency count in any category is too low (typically less than 5)
4. No Variable Interactions: Chi-square test, like other univariate feature selection methods, does not consider interactions between features. It might miss out on identifying important features that are significant in combination with other features

## Mutual Information:
It is a measure of the dependency between two variables. It quantifies the amount of information obtained about one random variable through observing the other random variable. It is a fundamental quantity in information theory.
Formula:
For discrete random variables X and Y: I(X;Y) = ∑∑ P(x,y) * log(P(x,y) / (P(x)P(y)))

Where P(x,y) is the joint probability distribution, and P(x) and P(y) are marginal probability distributions

### Mutual Information has several properties that make it useful for feature selection:
1. It is non-negative: MI is always zero or positive, with zero indicating that the variables are independent (i.e., no information about one variable can be obtained by observing the other variable)
2. It is symmetric: MI(X,Y) = MI(Y,X). The mutual information from X to Y is the same as from Y to X
3. It can capture any kind of statistical dependency: Unlike correlation, which only capture linear relationships mutual information can capture any kind of relationship, including nonlinear ones.

### Disadvantages:
1. Estimating Difficulty: Estimating MI from data can be challenging, especially when the dimensionality of the data is high or the number of samples is low. This is because MI estimation often relies on techniques like binning or density estimation, which can be sensitive to the chosen parameter or assumptions
2. Assumes Large Sample Sizes: MI works best with large sample sized. With smaller sample sizes, the estimates of MI can be noisy and less reliable, which might lead to incorrect conclusions about the dependencies between variables
3. Computationally Intensive: Calculating MI for many features can be computationally expensive. Especially for continuous variables. This might be problematic for large datasets or for applications where computational resources or time are limited
4. Difficulty with continuous variables: While MI theoretically applies to continuous variables, in practice it's often difficult to estimate MI between continuous variables due to the need for accurate density estimation, which is a challenging problem in its own right
5. No Direct Indication of the nature of Relationships: Although MI can identify the existence of a relationship between variables, it does not provide direct information about the nature of this relationship (e.g. linear, quadratic etc.) This contrasts with methods such as correlation, which directly indicate the strength and direction of a linear relationship
6. Doesn't Account for Redundancy: Mutual information measures the relevance of individual features to the target variable, but it doesn't take into account the redundancy among features. Two features might individually have high MI with the target, but if they are highly correlated, they might not provide much unique information. This can lead to the selection of redundant features. 


## Advantages and Disadvantages of filter based technique:

### Advantages:
1. Simplicity: Filter methods are generally straightforward and easy to understand. They involve calculating a statistic that measures the relevance of each feature, and selecting the top features based on this statistic
2. Speed: These methods are usually computationally efficient. Because they evaluate each feature independently, they can be much faster than wrapper methods or embedded methods, which need to train a model to evaluate feature importance
3. Scalability: Filter methods can handle a large number of features effectively because they don't involve any learning methods, This make them suitable for high-dimensional datasets
4. Pre-processing Step : They can serve as a pre-processing step for other feature selection methods. For instance, you could use a filter method to remove irrelevant features before applying a more computationally expensive method, such as a wrapper method.

### Disadvantages:
1. Lack of Feature Interaction: Filter methods treat each feature individually and hence do not consider the interactions between features. They might miss out on identifying important features that don't appear significant individually but are significant in combination with other features.
2. Model Agnostic: Filter methods are agnostic to the machine learning model that will be used for the prediction. This means that the selected features might not necessarily contribute to the accuracy of the specific model you want to use.


# Wrapper Methods:

Wrapper methods for feature selection are a type of feature selection methods that involve using a predictive model to score the combination of features. They are called "wrapper" methods because they "wrap" this type of model-based evaluation around the feature selection process.

Explanation: For features from f1,f2,f3…fn. We generate subsets. An example of subset can be (f1,f2,y). For every subset that is generated, we will apply ML algorithm and measure metrics (for example we could apply linear regression and apply R2 score for every subset.) The subset with the highest metric can be declared as the best features of our dataset.

### Here's how wrapper methods work in general:
1. Subset Generation: First, a subset of features is generated. This can be done in a variety of ways. For example, you might start with one feature and gradually add more, or start with all features and gradually remove them or generate subsets of features randomly. The subset generation method depends on the specific type of wrapper method being used.
2. Subset Evaluation: After a subset of features has been generated, a model is trained on this subset of features, and the model's performance is evaluated, usually through cross-validation. The performance of the model gives an estimate of the quality of the features in the subset.
3. Stopping Criterion: This process is repeated, generating and evaluating different subsets of features, until some stopping criterion is met. This could be a certain number of subsets evaluated, a certain amount of time elapsed, or no improvement in model performance after a certain number of iterations.

### Important techniques in Wrapper method:
1. Exhaustive feature selection 
2. Forward selection
3. Backward elimination selection
4. Recursive feature elimination


## Exhaustive feature selection: 
It involves systematically evaluating every possible combination of feature to identify the best subset for a given machine learning task
Documentation: https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/

### Process:
Start with the full set of N features
Generate all possible subsets of features (2^N - 1 combinations)
Evaluate each subset using a chosen performance metric
Select the subset that yields the best performance

### Disadvantage:
	1. Computational Complexity: The biggest drawback is its computational cost. If you have n features, the number of combinations to check is (2^n - 1), So, as the no of features grows, the no of combinations grows exponentially, making this method computationally expensive and time-consuming. For datasets with a large no of features, it may not be practical.
	2. Risk of Overfitting: By checking all possible combinations of features, there's a risk of overfitting the model to the training data. The feature combination that performs best on the training data may not necessarily perform well on unseen data
	3. Requires a Good Evaluation Metric: The effectiveness of exhaustive feature selection depends on the quality of the evaluation metric used to assess the goodness of a feature subset. If a poor metric is used, the feature selection may not yield optimal results.

## Backward elimination selection/Elimination:
It starts with all available features in the model, features are iteratively removed one at a time, this process is continued until a stopping criterion is met.

## Process:
	• Begin with a model containing all features
	• For each iteration:
		○ Remove each feature one at a time and evaluate model performance
		○ Identify the feature whose removal leads to the best model performance
		○ Permanently remove this feature from the model
	• Repeat until the stopping criterion is satisfied
### Disadvantage:
	• May not find the globally optimal feature subset (because we are not trying every possible option)
 
	
## Sequential Forward Selection:
It starts with an empty set of features and iteratively adds features one at a time, Features are added based on their contribution to model performance

### Process: 
	• Begin with an empty feature set
	• For each iteration:
		○ Evaluate the performance gain of adding each remaining feature
		○ Add the feature that provides the best performance improvement
		○ Remove this feature from the pool of available features
	• Repeat until a stopping criterion is met

### Disadvantage:
	• Can get stuck in local optima, missing potentially better feature combinations

## Recursive Feature Elimination:
It is a feature selection method that iteratively remove less important features to identify the most relevant features for a machine learning model.

### Process:
1. Train the model using all features
2. Rank features based on their importance (e.g. Coefficients for linear models , feature importance for tree-based models)
3. Remove the least important feature(s)
4. Retrain the model with the remaining features
5. Repeat steps 2-4 until the desired number of features is reached

### Implementation:
1. Available in scikit-learn as REF and RFECV ( RFE with cross-validation)



### Advantages and Disadvantages of using Wrapper method:

Advantages:
	1. Accuracy: Wrapper methods usually provide the best performing feature subset for a given machine learning algorithm because they use the predictive power of the algorithm itself for feature selection
	2. Interaction of Features: They consider the interaction of features. While filter methods consider each feature independently, wrapper methods evaluate subsets of features together. This means that they can find groups of features that together improve the performance of the model. Even if individually these features are not strong predictors.
Disadvantages:
	1. Computational Complexity: The main downside of wrapper methods is their computational cost. As they work by generating and evaluating many different subsets of features, they can be very time-consuming, especially for datasets with a large number of features.
	2. Risk of Overfitting: Because wrapper methods optimize the feature subset to maximize the performance of a specific machine learning model, they might select a feature subset that performs well on the training data but not as well on unseen data, leading to overfitting
	3. Model Specific: The selected feature subset is tailored to maximize the performance of the specific model used in the feature selection process. Therefore, this subset might not perform as well with a different type of model.

# Embedded Methods:

Embedded methods are feature selection techniques which perform feature selection as part of model construction process. They are called embedded methods because feature selection is embedded within the construction of the machine learning model. These methods aim to solve the limitations of filter and wrapper methods by including the interactions of the features while also being more computationally efficient.

There are certain machine learning algorithms that we can be able to use Embedded methods. Such as, algorithms that has "coef_" or "feature_importance_" as attributes:

![image](https://github.com/user-attachments/assets/8d73f7a9-b71e-4beb-b3af-35aa00d7004c)

We use sklearn "SelectFromModel" to do feature selection: 
Meta-transformer for selecting features based on importance weights.

From <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html> 

### Advantages and Disadvantages:

Advantages:
	1. Performance: They are generally more accurate than filter methods since they take the interactions between features into account
	2. Efficiency: They are more computationally efficient than wrapper methods since they fit the model only once
	3. Less Prone to Overfitting: They introduce some form of regularization, which helps to avoid overfitting. For example, Lasso and Ridge regression add a penalty to the loss function, shrinking some coefficients to zero
Disadvantages:
	1. Model Specific: Since they are tied to a specific machine learning model, the selected features are not necessarily optimal for other models
	2. Complexity: They can be more complex and harder to interpret than filter methods. For example, understanding why Lasso shrinks some coefficients to zero and not others can be non-trivial
	3. Tuning Required: They often have hyperparameters that need to be tuned, like the regularization strength in Lasso and Ridge regression.
	4. Stability: Depending on the model and the data, small changes in the data can result in different sets of selected features. This is especialy true for models that can fit complex decision boundaries, like decision trees.


## Cheat sheet:

1. Filter Methods:
	1. Variance Threshold: Removes all features whose variance doesn't meet a certain threshold. Use this when you have many features and you want to remove those that are constants or near constants
	2. Correlation Coefficient : Finds the correlation between each pair of features. Highly correlated features can be removed since they contain similar information. Use this when you suspect that some features are highly correlated
	3. Chi-Square Test: This statistical test is used to determine if there's a significant association between two variables. It's commonly used for categorical variables. Use this when you have categorical features and you want to find their dependency with the target variable
	4. Mutual Information: Measures the dependency between two variables. It’s a more general form of the correlation coefficient and can capture non-linear dependencies. Use this when you want to measure both linear and non-linear dependencies between features and the target variable
	5. ANOVA (Analysis of Variance): ANOVA tests the impact of one or more factors by comparing the means of different samples. Use this when you have one or more categorical independent variables and a continuous dependent variable.
2. Wrapper Methods:
	1. Recursive Feature Elimination (RFE): Recursively removes features, builds a model using the remaining attributes, and calculates model accuracy. It uses  model accuracy to identify which attributes contribute the most. Use this when you want to leverage the model to identify the best features.
	2. Sequential Feature Selection (SFS): Adds or removes one feature at the time based on the classifier performance until a feature subset of the desired size k is reached. Use this when computational cost is not an issue and you want to find the optimal feature subset.
	3. Exhaustive Feature Selection: This is a brute-force evaluation of each feature subset. This method, as the name suggests, tries out all possible combinations of variables and returns the best subset. Use this when the number of features is small , as it can be computationally expensive.
3. Embedded Methods:
	1. Lasso Regression: Lasso ( Least Absolute Shrinkage and Selection Operator) is a regression analysis method that performs both variable selection and regularization. Use this when you want to create a simple and interpretable model.
	2. Ridge Regression: Ridge regression is a method used to analyze multiple regression data that suffer from multicollinearity. Unlike Lasso, it doesn't lead to feature selection but rather minimizes the complexity of the model.
	3. Elastic Net: This method is a combination of Lasso and Ridge. It incorporates penalties from both methods and is particularly useful when there are multiple correlated features.
	4. Random Forest Importance: Random forests provide a straightforward method for feature selection, namely mean decrease impurity (MDI). Use this when you want to leverage the power of random forests for feature selection. 

