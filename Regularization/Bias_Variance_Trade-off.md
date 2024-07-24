Reducible error = Bias^2 + variance

![image](https://github.com/user-attachments/assets/de333427-c1f4-480c-bba0-2ebc5fe4d43b)

## Bias:
Bias = The inability of a Machine Learning model to fit the training data

## Variance:
Variance: Change in Machine Learning model's prediction when the training data is changed

High variance is closely related to Over fitting
High Bias is highly related to Underfitting

Ideally what we need is Low Bias and Low Variance

![image](https://github.com/user-attachments/assets/bef8c337-268a-43ea-8739-aa4ddfc854ca)

Explanation: Low Bias means, the process we do to train the model perfectly for the given training data
For example, if we split our data set to X_train and X_test. The process of getting our model to fit all the training data will result in LOW BIAS.
If we then give our model with new data sets, (here let's assume X_test) the model might or might not show a good fit. If it show a good fit it is said to be low variance and if not it is said to have high variance. If we try to introduce multiple test datasets for our model, the differences it shows is called Variance. Usually out aim is to keep the model with low bias and low variance. But in reality as we try to reduce the bias we introduce high variance. This is called trade-off between bias and variance.

The "trade-off" in bias-variance trade-off refers to the fact that minimizing bias will usually increase variance and vice versa.

Some question to answer:
1. How would you define bias and variance mathematically ?
2. How is bias and variance related to overfitting and underfitting mathematically ?
3. Why is there a tradeoff between bias and variance mathematically ?

To answer these questions we need to first know about Expected value and variance:

Expected value represents the average outcome of a random variable over a large number of trails or experiments.
In a simple sense, the expected value of a random variable is the long-term average value of repetition of the experiment it represents. For example, the expected value of rolling a six - sided die is 3.5 because, over many rolls we would expect to average about 3.5

Expected value is nothing but roughly we can say it is population mean

![image](https://github.com/user-attachments/assets/6c8173ab-5e0e-4f1a-8ba2-b09a08a6607d)

If we have a discrete random variable (x) , 
expected value E(x) = ∑ i = 1 to n xi (pxi)


If we have a Continious random variable (x) , 
expected value E(x) = ∫ xi f(xi) dx 

![image](https://github.com/user-attachments/assets/a7474612-02f6-489d-883b-309409cd285e)

### What exactly are Bias and Variance Mathematically?

In the context of machine learning and statistics, bias refers to the systematic error that a model introduces because it cannot capture the true relationship in the data. 
It represents the difference between the expected prediction of our model and the correct value which we are trying to predict. More bias leads to underfitting, where the model does not fit the training data well.

Bias = E[f'(x)] - f(x)

![image](https://github.com/user-attachments/assets/7b558849-2a87-4b0a-a641-d08775fc944e)

Bias =  E[m] - m , when E[m] is m and when m - m = 0 then bias = 0. When bias is 0 we call out machine learning as unbiased predictor.

In the context of machine learning and statistics, variance refers to the amount by which the prediction of our model will change if we used a different training data set. 
In other words, it measures how much the predictions for a given point vary between different realizations of the model. 

Variance Var(f'(x)) = E[(f ' (x) - E [ f' (x)])^2

![image](https://github.com/user-attachments/assets/22f0d82a-4793-4445-bfc8-c441e75a99c8)

If variance is high then it is overfitting.


### Bias Variance Decomposition:
Bias-variance decomposition is a way of analyzing a learning algorithm's expected generalization error with respect to a particular problem by expressing it as the sum of three very different quantities: bias, variance, and irreducible error.

1. Bias: This is the error from erroneous assumptions in the learning algorithm. High bias can accuse an algorithm to miss the relevant relations between features and target outputs (underfitting).
2. Variance: This is the error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).
3.Irreducible Error: This is the noise term. This part of the error is due to the inherent noise in the problem itself, and can't be reduced by any model.











