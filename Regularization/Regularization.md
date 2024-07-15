# Q. What is Regularization ?

Regularization is a set of techniques used in machine learning to prevent overfitting and improve a model's ability to generalize to new, unseen data.
It reduces overfitting by adding a penalty to the model's loss function. It improves generalization by controlling model complexity. It balances the trade-off between bias and variance.
## Types of Regularizations:

1. Ridge
2. Lasso
3. ElasticNet

![image](https://github.com/user-attachments/assets/f133944d-3dd1-4dd8-a6c2-64c860f7549d)

## When to use Regularization?
1. Preventing Overfitting: Regularization is most commonly used as a tool to prevent overfitting. OF your model performs well on the training data but poorly on the validation or test data, it might be overfitting, and regularization could help
2. High Dimensionality: Regularization is particularly useful when you have a high number of features compared to the number of data points. In such scenarios, models tend to overfit easily and regularization can help by effectively reducing the complexity of the model
3. Multicollinearity: When features are highly correlated(multicollinearity), it can destabilize your model and make the model's estimates sensitive to minor changes in the model. L2 regularization (Ridge regression) can help in such cases by distributing the coefficient estimates among correlated features
4. Feature Selection: If you have a lot of features and you believe many of them might be irrelevant, L1 regularization (Lasso) can help. It tends to produce sparse solutions, driving the coefficients of irrelevant features to zero, thus performing feature selection
5. Interpretability: If model interpretability is important and you want a simpler model, regularization can help achieve this by constraining the model's complexity
6. Model Performance: Even if you're not particularly worried about overfitting, regularization might still improve your model's out-of-sample prediction performance.

---
- Let's try to understand Overfitting in terms of Linear Regression:

In Overfitting, a model learns the training data too well, including its noise and random fluctuations resulting in poor performance on new, unseen data.  The model try's to memorize the training the data instead of learning to generalize.

The whole idea of Linear Regression is to find out the best fit line y=mx+b
Coefficients in y=mx + b represents the weightage of X in finding the value of y. If m value is so high that means, to find out the value of y , X has a lot of importance or weightage. Same way if the m value is less that implies that x importance or weightage to find out the value of y is low.

If we observe the overfit models we usually notice that the coefficients have very high values.
So, if we talk overfitting in terms of Linear Regression, the value of m is extremely high and if the value of m is very low that means that model is underfitting.

Now if there is overfitting in the model we can clearly see that the slope value is high and to reduce the overfitting we need to reduce the slope.

For example: if I split a dataset to test and training data, my training data has less points and the best fit line is passing through those lines. Because of less data in training dataset, the model failed to capture the pattern of the entire dataset.  We can clearly observe the best fit line in the  below image, it is showing overfitting  but if we look at the green line it is capturing most of the data patterns, if we use this green line although it make errors on the training data it performs good on the test data. So, we need to convince ML model to use green line instead of blue line. 
This can be done using regularization.

![image](https://github.com/user-attachments/assets/1884b119-b8e3-4fd1-83b7-448be3363f37)

Firstly how do we select the line ? 
By considering error minimization, we do MSE to calculate loss function and we do it either by doing OLS or Gradient descent.

L = ∑ I = 1 to n (yi - y^i)^2

In regularization, we will add an extra term to the loss function 
Extra term is lambda(m^2)
So, in regularization the loss function is:

∑ I = 1 to n (yi - y^i)^2   + (Lambda)(m^2)
Where:
Lambda, is a hyperparameter, where we can tweak the parameter value ranging 0 to ∞.
 m = slope
And if we have more than one input column, then we add those m values as well to the lambda equation.
Lamba(m1^2 + m2^2+ ----mn^2)
Since we are using squares for our coefficients in our lambda equation this is called as L2 norm.
In L1 norm we use mod |m| instead of squares.

Using this new formula we need to calculate the loss function of both the curves and let ML model choose the curve with low loss function value.

