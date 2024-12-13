#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#importing dataset
df = pd.read_csv('dataset.csv')
df.head()


# In[3]:


#1 a. Number of Instances

print("Number of Instances:", df.shape[0])


# In[4]:


#1 b. Number of Features

print("Number of Features:", len(df.drop(columns= 'classification').columns))


# In[5]:


#1 c. Number of instances from each class (malware/benign)

print("Number of instances from each class:", df['classification'].value_counts())


# In[6]:


#Checking Datatypes for each column

df.info()


# In[7]:


#Check for Null Values

df.isnull().sum()


# In[8]:


#Check for Duplicates

df.duplicated().sum()


# In[9]:


#check number of rows and columns

df.shape


# In[10]:


#Checking Outliers

for i in df.columns:
  if df[i].dtypes != object:
    sns.boxplot(df[i])
    plt.title(i)
    plt.show()


# In[11]:


#Removing Outliers

col_names=['vm_truncate_count','mm_users','map_count', 'total_vm', 'exec_vm','reserved_vm','nvcsw','nivcsw', 'utime','gtime','fs_excl_counter']

for i in col_names:
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each column
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    
    # Apply IQR filtering to remove outliers
    df = df[(df[i] >= Q1 - 1.5 * IQR) & (df[i] <= Q3 + 1.5 * IQR)]


# In[12]:


#Statistical summary of the data
df.describe()


# In[13]:


# Checking for colrelation between columns. Heatmap

plt.figure(figsize=(30,20))
sns.heatmap(df.corr(numeric_only = True), annot = True)


# Columns 'maj_fit', 'end_data', and 'shared_vm' are linearly dependent.

# In[15]:


# Data preprocessing

# Label Encoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in df.columns:
  if df[i].dtypes == "object":
    df[i] = le.fit_transform(df[i])


# In[16]:


df.head()


# In[17]:


#Applying PCA to reduce dimmension after scaling

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Separate features from the target variable if applicable

X = df.drop('classification',axis=1).to_numpy()
y = df['classification'].to_numpy()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# 3. Explained Variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# 4. Visualize PCA components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.colorbar(label='Target Class')
plt.show()


# In[18]:


#2. Load and split the dataset into two parts (training and testing)

#Model Building

#Create Train and Test datasets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y,test_size = 0.20,random_state=100)


# In[19]:


# Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


x_train2 = X_train
x_test2 = X_test


for name,method in [('Logistic Regression', LogisticRegression(solver='liblinear',random_state=100))]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    target_names=['0','1']
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict,target_names=target_names))


#Confusion matrix Plot:
    cm = confusion_matrix(y_test, predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# In[20]:


# Decision Tree Model

x_train2 = X_train
x_test2 = X_test

#Script for Decision Tree
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

for name,method in [('Decision Tree', DecisionTreeClassifier(random_state=100))]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    target_names=['0','1']
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict,target_names =target_names))

#Confusion matrix Plot:

    cm = confusion_matrix(y_test, predict)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# In[21]:


# Random Forest Model

#Script for Random forest
from sklearn.ensemble import RandomForestClassifier  

for name,method in [('RF', RandomForestClassifier(n_estimators = 100, random_state=100))]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    target_names=['0','1']
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict,target_names=target_names))

#Confusion matrix Plot:

    cm = confusion_matrix(y_test, predict)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# In[22]:


## Neural Networks

#Building Neural Network model
from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(5,4,5),max_iter=10000,random_state=100)  
mlp.fit(x_train2, y_train) 
predictions = mlp.predict(x_test2) 

#Evaluation Report and Matrix
from sklearn.metrics import classification_report, confusion_matrix  
target_names=['0','1']
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions,target_names=target_names))

#Confusion matrix Plot:

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Neural Network - Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[55]:


## kNN Model
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time



k_range = range(1, 10)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, x_train2, y_train, cv=10, scoring='accuracy')
    #append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())
    
#Print Best Score (i.e K value)
BestScore = [1 - x for x in k_scores]
best_k = k_range[BestScore.index(min(BestScore))]
print('Optimal Number of Neighbors: {}'.format(best_k))

start_bs = time.time()

#Create Optimal KNN model
classifier = KNeighborsClassifier(n_neighbors = best_k)
from sklearn.metrics import classification_report, confusion_matrix  

for name,method in [('k-NN', classifier)]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict))


end_bs = time.time()

end_bs = time.time()  # End the timer
execution_bs = end_bs - start_bs
print(f"Execution time: {execution_bs:.4f} seconds")


#Confusion matrix Plot:

cm = confusion_matrix(y_test, predict)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title(f'{name} - Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[53]:


#AUC-ROC Curve for Log reg and kNN
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

logistic_model = LogisticRegression(solver='liblinear', random_state=100)
logistic_model.fit(x_train2, y_train)
logistic_pred = logistic_model.predict(x_test2)
logistic_prob = logistic_model.predict_proba(x_test2)[:, 1]

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(x_train2, y_train)
knn_pred = knn_model.predict(x_test2)
knn_prob = knn_model.predict_proba(x_test2)[:, 1]

# AUC-ROC for Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, logistic_prob)
auc_lr = auc(fpr_lr, tpr_lr)

# AUC-ROC for k-NN
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_prob)
auc_knn = auc(fpr_knn, tpr_knn)

# Plot AUC-ROC for Both Models
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})', color='blue')
plt.plot(fpr_knn, tpr_knn, label=f'k-NN (AUC = {auc_knn:.2f})', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.title('AUC-ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# In[ ]:




