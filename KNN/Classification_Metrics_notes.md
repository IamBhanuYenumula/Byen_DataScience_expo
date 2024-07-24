### Accuracy: 
Accuracy score is the fraction of predictions that a classification model got right. It's calculated by dividing the number of correct predictions by the total number of predictions.

### Accuracy of multi-classification:
Accuracy measures the proportion of correctly classified instances out of the total instances. The formula for accuracy in multi-class classification is similar to that in binary classification:
Accuracy=Number of correct predictions/Total number of predictions

### How much accuracy is good ?
It depends on the problem that we are solving and the data we got to model

### What is the problem with Accuracy?
Accuracy score will not tell us the type of the error. For instance: If we get 90% accuracy for a model, we don't know the type of error we have for the remaining 10%. 

### What is Confusion Matrix ?

![image](https://github.com/user-attachments/assets/01c748e4-3e87-4fe7-86c9-2fec16be43e8)

A confusion matrix is a tabular summary of a classifier's predictions compared to the actual class labels, showing true positives, true negatives, false positives, and false negatives.

### What is Type 1 Error and Type ll error ?

Type 1: 
A type 1 error occurs when a null hypothesis is rejected even though it is actually true. It's also known as a "false positive".

Type ll: 
A Type II error, also known as a false negative, occurs when a hypothesis test fails to reject a null hypothesis that is actually false. 


### Confusion Matrix for Multi-classification Problem:
A confusion matrix for a multi-class classification problem is a square matrix where the rows represent the actual classes and the columns represent the predicted classes. Each cell in the matrix indicates the number of instances where the actual class is the row and the predicted class is the column.
For a classification problem with 𝑛n classes, the confusion matrix will be an 𝑛×𝑛n×n matrix. The diagonal elements represent the number of instances correctly classified for each class, while the off-diagonal elements represent misclassifications.


### When will accuracy be misleading ?
When we have Imbalance dataset the accuracy will be misleading. 
When classes are not equally represented in the dataset, accuracy can be misleading. A model might achieve high accuracy by simply predicting the majority class most of the time, while performing poorly on minority classes.

To address these limitations, it's often recommended to use additional metrics such as precision, recall, F1-score, and confusion matrices, which provide a more comprehensive view of the model's performance across all classes. These metrics can help identify specific areas where the model needs improvement and provide a more nuanced understanding of its strengths and weaknesses.



### What is Precision:
Precision is a metric used in classification problems to measure the accuracy of the positive predictions made by a model. It is particularly useful in scenarios where the cost of false positives is high.
Precision=True Positives TP/True Positives TP +False Positives FP  ​


### Precision in Multi-Class Classification
In multi-class classification, precision can be calculated for each class individually. The overall precision can then be averaged using:
Macro-Averaging: Calculates the precision independently for each class and then takes the average.
Micro-Averaging: Aggregates the contributions of all classes to compute the average precision.

### What is Recall ?
Recall measures the proportion of actual positive cases that were correctly identified by the model.
Formula:
Recall = True Positives / (True Positives + False Negatives)

### What is F1 Score?
The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both.
Formula:
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
Range: The F1 score ranges from 0 to 1, with 1 being the best possible score and 0 the worst.

### Harmonic mean: 
The harmonic mean is the reciprocal of the arithmetic mean of reciprocals of a given set of numbers.
Properties of Harmonic mean:
	• Always lower than or equal to the arithmetic mean
	• Gives more weight to smaller values in a dataset
	• Appropriate for averaging rates or speeds


### How to calculate Precision , Recall and F1 in multi-class classification?

### For Precision:
1. Macro Precision = average of all variables precision
2. Weighted Precision = each precision value is multiplied with its respective class weight
### For Recall:
1. Macro Recall
2. Weighted Recall
### For F1 score:
1. Macro F1 score
2. Weighted F1 score

NOTE: We can use classification report from sklearn.metrics, this will give complete report of the metrics with each variable. It also gives us Macro and Weighted scores of each variable.
