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

## key understandings about Ridge regression:
1. How the coefficients get affected
    ○ When we add lambda value (which will be between 0 - α ), all the coefficients will be shrink towards zero, but the values will never be zero. 
2. Higher values are impacted
    ○ Coefficients with higher values will shrink more compared to coefficients with less values
3. Bias variance tradeoff
    ○ Bias and Variance depends on lambda value: 
        § If lambda value is close to 0, model will show a Overfit tendency where you will notice a decrease in bias and increase in variance
        § If lambda value is high, model will show Underfit tendency where you notice a decrease in Variance but an increase in Bias
4. Impact on the Loss function
    ○ By increasing lambda value the loss function tends to move towards zero and shrink its size. This phenomenon explains why coefficients tends towards zero when lambda value is increased
5. Why called Ridge
    Hard constraint Ridge regression: Because the solution always lies on the ridge of the circle, that we get when we plot the lambda term in the ridge loss function

NOTE: Always try using Ridge function when there are 2 or more than 2 input columns


Lass Regression:
Lasso Regression is very similar to Ridge regression, here we use "MSE" as loss function and  we will add a penalty term.  And the penalty term is "l1" norm. 

Therefore the equation is : L = MSE + lambda ||w||

Where:
||W|| = absolute value of |w1| + |w2|+…+|Wn|

## Lasso regression's advantage:
If we are working on a high dimensional data, for example: x1,x2,x3…..xn. The chances of overfitting in high dimensional data is very high. If we apply Ridge regression on this data the coefficients will not become zero. Whereas, if we apply Lasso regression for this dataset and if we increase lambda value sightly, it will make the less weights coefficients as zero. This means that Lasso regression is inherently performing feature selection. 
In other words, Lasso regression is letting us know the less important features in the dataset and by making their coefficients value as zero it is performing dimensionality reduction. 

## key understandings about Lasso regression:
1. How the coefficients get affected
        ○ When we add lambda value (which will be between 0 - α ), all the coefficients will be shrink to zero. 
2. Higher values are impacted
        ○ Coefficients with higher values will shrink more compared to coefficients with less values and increasing alpha value will eventually zero out all features
3. Bias variance tradeoff
○ Bias and Variance depends on lambda value: 
§ If lambda value is close to 0, model will show a Overfit tendency where you will notice a decrease in bias and increase in variance
§ If lambda value is high, model will show Underfit tendency where you notice a decrease in Variance but an increase in Bias
4. Impact on the Loss function
By increasing lambda value the loss function tends to move to zero and we lose the shape of the curve as well