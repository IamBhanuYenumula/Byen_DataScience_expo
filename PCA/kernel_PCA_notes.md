Kernel PCA comes into picture when PCA fails. PCA assumes that the principal components are a linear combination of the original features. It can't handle complex polynomial relationships between features.
Kernel PCA extends the capabilities of PCA by using kernel methods to perform the operations in a high-dimensional feature space. This approach allows Kernel PCA to uncover hidden non-linear features within complex data.

In Kernel PCA, data is first transformed to higher dimension and it is again put back to lower dimension to separate the data points. For example, if the data points are in 2D, we first take them to 3D and apply a principal component and project It back to 2D to separate the data points. 

![image](https://github.com/user-attachments/assets/1c3b58f2-1c16-4331-9e18-5df6d4cf5809)

