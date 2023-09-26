#!/usr/bin/env python
# coding: utf-8

# # 1.Retrieve and load the mnist_784 dataset of 70,000 instances.

# In[1]:


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import IncrementalPCA


# In[2]:


mnist = fetch_openml('mnist_784')


# In[3]:


X, y = mnist.data, mnist.target


# In[4]:


type(y[1])


# In[5]:


# y was in str so converted it into int
y = y.astype(int)


# In[6]:


print("Shape of the feature :", X.shape)
print("Shape of the target :", y.shape)


# # 2. Display each digit.

# In[7]:


for i in range(10):  
    digit_data = X.iloc[i].to_numpy().reshape(28, 28)  # Reshape the 1D vector to a 2D image
    plt.figure(figsize=(1, 1))
    plt.imshow(digit_data, cmap='gray')
    plt.title(f"Digit: {y[i]}")
    plt.axis('off')
    plt.show()


# # 3. Use PCA to retrieve the 1st and 2nd principal component and output their explained variance ratio.

# In[8]:


n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Get the 1st and 2nd principal components
pc1 = pca.components_[0]
pc2 = pca.components_[1]


# # 4. Plot the projections of the 1st and 2nd principal component onto a 1D hyperplane.

# In[9]:


# Create a 1D hyperplane for the 1st principal component
projection_1st = np.dot(X, pc1)

# Create a 1D hyperplane for the 2nd principal component
projection_2nd = np.dot(X, pc2)

# Plot the projections
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(projection_1st, np.zeros_like(projection_1st), c='b', alpha=0.5)
plt.title('Projection onto 1st Principal Component')
plt.xlabel('Projection Value')
plt.ylabel('Arbitrary Y-axis')

plt.subplot(122)
plt.scatter(projection_2nd, np.zeros_like(projection_2nd), c='r', alpha=0.5)
plt.title('Projection onto 2nd Principal Component')
plt.xlabel('Projection Value')
plt.ylabel('Arbitrary Y-axis')

plt.tight_layout()
plt.show()


# # 5.Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions. 

# In[10]:


# Specify the desired number of dimensions (154 in this case)
n_components_r = 154

# Create an IncrementalPCA object
ipca = IncrementalPCA(n_components=n_components_r)

# Fit IPCA to the data
X_ipca = ipca.fit_transform(X)

# Check the shape of the reduced data
print("Shape of X after IPCA:", X_ipca.shape)


# In[11]:


#Display the original and compressed digits 
sample_indices = np.random.choice(X_ipca.shape[0], 10, replace=False)

plt.figure(figsize=(12, 5))

#This will plot the original data
plt.subplot(1, 2, 1)
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 10, i + 1)
    plt.imshow(ipca.inverse_transform(X_ipca[idx]).reshape(28, 28), cmap='gray')
    plt.title(f'Original')
    plt.axis('off')

#This will plot the compressed data
plt.subplot(1, 2, 2)
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 10, i + 11)
    plt.imshow(ipca.inverse_transform(X_ipca[idx]).reshape(28, 28), cmap='gray')
    plt.title(f'Compressed')
    plt.axis('off')

plt.tight_layout()
plt.show()


# # Question 2

# # 1. Generate Swiss roll dataset. 
# # 2. Plot the resulting generated Swiss roll dataset.
# 

# In[12]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


n_samples = 1000  # Number of data points
noise = 0.2      # Amount of noise to add
X, color = make_swiss_roll(n_samples=n_samples, noise=noise)

# Plot the Swiss roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the data points with color based on the unrolled angle
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Swiss Roll Dataset")
plt.show()


# # 3. Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points).
# # 4.Plot the kPCA results of applying the linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points) from (3). Explain and compare the results 

# In[13]:


#Below is Kernel PCA (kPCA) is used with linear kernel 
kpca_linear = KernelPCA(kernel="linear", n_components=2)

#Below is Kernel PCA (kPCA) is used with a RBF kernel
kpca_rbf = KernelPCA(kernel="rbf", gamma=0.04, n_components=2)

#Below is Kernel PCA (kPCA) is used with a sigmoid kernel.
kpca_sigmoid = KernelPCA(kernel="sigmoid", gamma=0.001, n_components=2)

X_linear = kpca_linear.fit_transform(X)
X_rbf = kpca_rbf.fit_transform(X)
X_sigmoid = kpca_sigmoid.fit_transform(X)

plt.figure(figsize=(15, 5))

#The below will plot the linear kernel graph
plt.subplot(131)
plt.scatter(X_linear[:, 0], X_linear[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with Linear Kernel")

#The below will plot the RBF kernel graph
plt.subplot(132)
plt.scatter(X_rbf[:, 0], X_rbf[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with RBF Kernel")

#The below will plot the Sigmoid kernel graph
plt.subplot(133)
plt.scatter(X_sigmoid[:, 0], X_sigmoid[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("kPCA with Sigmoid Kernel")

plt.tight_layout()
plt.show()


# # 5. Using kPCA and a kernel of your choice, apply Logistic Regression for classification. Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at the end of the pipeline. Print out best parameters found by GridSearchCV. 

# In[15]:


from sklearn.preprocessing import StandardScaler

n_samples = 1000
X, color = make_swiss_roll(n_samples, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, color, test_size=0.2, random_state=42)

threshold = np.median(color)  
y_train_binary = (y_train > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('kpca', KernelPCA(kernel='rbf')),  
    ('classifier', LogisticRegression(max_iter=1000))
])

param_grid = {
    'kpca__kernel': ['rbf', 'sigmoid', 'poly'],  
    'kpca__gamma': [0.001, 0.01, 0.1, 1.0, 10.0]  
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train_binary)

best_params = grid_search.best_params_
print("Best Parameters:")
print(best_params)

best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test)

#The below will print the accuracy score
accuracy = accuracy_score(y_test_binary, y_pred)
print(f"Accuracy on Test Set: {accuracy:.2f}")


# # 6. Plot the results from using GridSearchCV in (5).

# In[16]:


scores = grid_search.cv_results_["mean_test_score"]
gammas = [params["kpca__gamma"] for params in grid_search.cv_results_["params"]]
plt.figure(figsize=(10, 6))
plt.scatter(gammas, scores, c=scores, cmap=plt.cm.viridis)
plt.colorbar(label="Mean Test Score")
plt.xlabel("Gamma")
plt.ylabel("Mean Test Score")
plt.title("GridSearchCV Results")
plt.show()


# In[ ]:




