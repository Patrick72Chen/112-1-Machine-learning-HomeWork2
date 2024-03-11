#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 00:03:07 2023

@author: chenmaige
"""

#1 intall sklearn
#conda install scikit-learn=0.24.2



#2 import套件
import pandas as pd
import numpy as np
import re
import sklearn
from sklearn.utils import resample

#for normalization
from sklearn.preprocessing import StandardScaler

#For Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import itertools

#For Metrics evaluation 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



#3 導入資料
df = pd.read_csv('/Users/chenmaige/Downloads/archive/train.csv')#更改為資料位置路徑



#4 檢查遺失值、丟掉policy_id colum
df.isna().sum() 
df = df.drop('policy_id', axis=1)



#5 畫圖檢視數據是否平衡
fig = plt.figure(facecolor='white')

ax = fig.add_subplot(1, 1, 1, facecolor='white')

plt.pie(df['is_claim'].value_counts(), labels=['No Claim', 'Claim'], radius=1, colors=['green', 'orange'],
        autopct='%1.1f%%', explode=[0.1, 0.15], labeldistance=1.15, startangle=45,
        textprops={'fontsize': 15, 'fontweight': 'bold'})

plt.legend(title='Outcome:', loc='upper right', bbox_to_anchor=(1.6, 1))

fig.patch.set_facecolor('white')

plt.show()



#6 換成torque/rpm、換成power/rpm
df['torque'] = df['max_torque'].apply(lambda x: re.findall(r'\d+\.?\d*(?=Nm)', x)[0])
df['rpm'] = df['max_torque'].apply(lambda x: re.findall(r'\d+\.?\d*(?=rpm)', x)[0])

df['torque'] = pd.to_numeric(df['torque'])
df['rpm'] = pd.to_numeric(df['rpm'])

df['torque_to_rpm_ratio'] = df['torque'] / df['rpm']

df.drop('max_torque', axis=1,inplace=True)
df.drop('rpm',axis=1,inplace=True)
df.drop('torque',axis=1,inplace=True)


df['power'] = df['max_power'].apply(lambda x: re.findall(r'\d+\.?\d*(?=bhp)', x)[0])
df['rpm'] = df['max_power'].apply(lambda x: re.findall(r'\d+', x)[-1])

df['power'] = pd.to_numeric(df['power'])
df['rpm'] = pd.to_numeric(df['rpm'])

df['power_to_rpm_ratio'] = df['power'] / df['rpm']

df.drop('power', axis=1,inplace=True)
df.drop('rpm',axis=1,inplace=True)
df.drop('max_power',axis=1,inplace=True)



#7 把yes/no換成0,1、檢查數字行變數及物件變數
is_cols=[col for col in df.columns if col.startswith("is") and col!="is_claim"]
print(is_cols)

df = df.replace({ "No" : 0 , "Yes" : 1 })


#數字型變數
dataset_num_col = df.select_dtypes(include=['int', 'float']).columns
print(" Data Set Numerical columns:")
print(dataset_num_col.nunique())
print(dataset_num_col)

#物件變數
dataset_cat_cols = df.select_dtypes(include=['object']).columns
print("Data Set categorical columns:")
print(dataset_cat_cols.nunique())
print(dataset_cat_cols)

#將物件變數轉換為二進制形式
df= pd.get_dummies(df, columns=dataset_cat_cols,drop_first=True)




#8處理數據不平衡
majority_class = df[df['is_claim'] == 0]
minority_class = df[df['is_claim'] == 1]

undersampled_majority = resample(
    majority_class,
    replace=False,  
    n_samples=len(minority_class) * 2,  
    random_state=42  
)

# Combine the undersampled majority class with the minority class
df_final = pd.concat([undersampled_majority, minority_class])



#9 標準化連續型變數
#標準化
#population_density
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['population_density']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['population_density'] = column_normalized

#標準化
#displacement
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['displacement']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['displacement'] = column_normalized

#標準化
#length
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['length']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['length'] = column_normalized

#標準化
#width
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['width']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['width'] = column_normalized

#標準化
#height
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['height']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['height'] = column_normalized

#標準化
#gross_weight
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['gross_weight']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['gross_weight'] = column_normalized




#10 分好X, y，並將資料切分
X = df_final.drop('is_claim', axis = 1)
y = df_final['is_claim']


#train=8:test=2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)



#11 繪製圓餅圖
class_counts = np.bincount(y)
plt.pie(class_counts, labels=['No Claim', 'Claim'], radius=1.5, colors=['#FFFACD', '#ADD8E6'],
        autopct='%1.1f%%', labeldistance=1.15, startangle=0)
plt.show()


#前處理結束







############################################第一大題#####################################################################


#Kernelized percentron

class PerceptronClassifier:
    def __init__(self, learning_rate=0.1, epochs=100, shuffle=True, gamma=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.shuffle = shuffle
        self.gamma = gamma

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None:
            X_val = self.rbf_kernel(X_val, X_train)
        self.X_train = X_train.astype(float)  # 確保 X_train 的類型是 float
        X_train_rbf = self.rbf_kernel(self.X_train, self.X_train)

        num_features = X_train_rbf.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            if self.shuffle:
                shuffled_indices = np.random.permutation(len(X_train_rbf))
                X_train_rbf = X_train_rbf[shuffled_indices]
                y_train = y_train[shuffled_indices]

            for i in range(len(X_train_rbf)):
                prediction = np.dot(X_train_rbf[i], self.weights) + self.bias
                y_pred = 1 if prediction >= 0 else 0

                if y_pred != y_train[i]:
                    self.weights += self.learning_rate * (y_train[i] - y_pred) * X_train_rbf[i]
                    self.bias += self.learning_rate * (y_train[i] - y_pred)

            train_accuracy = np.mean(self.predict(self.X_train) == y_train)
            print(f"Epoch {epoch + 1}/{self.epochs} - Training Accuracy: {train_accuracy}")

    def rbf_kernel(self, X1, X2):
         pairwise_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
         return np.exp(-self.gamma * pairwise_dists)
    
    def rbf_kernel(self, X1, X2):
        X1 = X1.astype(float)  
        X2 = X2.astype(float)  

        # Reshape X1 and X2 if they are 1D arrays
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        pairwise_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * pairwise_dists)


    def predict(self, X_test):
        X_test_rbf = self.rbf_kernel(X_test, self.X_train)
        y_pred_perceptron = np.where(np.dot(X_test_rbf, self.weights) + self.bias >= 0, 1, 0)
        return y_pred_perceptron

    def evaluate(self, X_test, y_test):
        X_test = X_test.astype(float)  # 確保 X_test 的類型是 float
        X_test_rbf = self.rbf_kernel(X_test, self.X_train)
        y_pred_perceptron = self.predict(X_test)
        accuracy_perceptron = np.mean(y_pred_perceptron == y_test)
        return accuracy_perceptron, y_pred_perceptron, self.weights, self.bias


# 宣告模型
perceptron_model = PerceptronClassifier(learning_rate=0.1, epochs=100, shuffle=True, gamma=1.0)

# 訓練模型
perceptron_model.fit(X_train, y_train)

# 评估模型
accuracy, predictions, weights, bias = perceptron_model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
print(f"Predictions: {predictions}")
print(f"Weights: {weights}")
print(f"Bias: {bias}")



#Kernelized K-NN

class KernelizedKNNClassifier:
    def __init__(self, k=3, gamma=1.0):
        self.k = k
        self.gamma = gamma
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train.astype(float)  # Ensure X_train's type is float
        self.y_train = y_train

    def rbf_kernel(self, X1, X2):
        pairwise_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * pairwise_dists)

    def predict(self, X_test):
        X_test = X_test.astype(float)  # Ensure X_test's type is float
        X_test_rbf = self.rbf_kernel(X_test, self.X_train)

        y_pred_knn = []
        for i in range(X_test_rbf.shape[0]):
            dists = np.sum((X_test_rbf[i] - self.rbf_kernel(self.X_train, self.X_train))**2, axis=1)
            neighbors_indices = np.argsort(dists)[:self.k]
            neighbor_labels = self.y_train[neighbors_indices]

            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            y_pred_knn.append(predicted_label)

        return np.array(y_pred_knn)

    def evaluate(self, X_test, y_test):
        X_test = X_test.astype(float)  # Ensure X_test's type is float
        y_pred_knn = self.predict(X_test)
        accuracy_knn = np.mean(y_pred_knn == y_test)
        return accuracy_knn, y_pred_knn


# Declare the model
knn_model = KernelizedKNNClassifier(k=3, gamma=1.0)

# Train the model
knn_model.fit(X_train, y_train)

# Evaluate the model
accuracy_knn, predictions_knn = knn_model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy_knn}")
print(f"Predictions: {predictions_knn}")



#k-NN(manhattan_distance)

class KNN_manhattan_Classifier:
    def __init__(self, k=3):
        self.k = k

    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []

        for test_point in X_test:
            distances = [self.manhattan_distance(test_point, train_point) for train_point in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            predicted_label = 1 if sum(nearest_labels) >= self.k / 2 else 0
            y_pred.append(predicted_label)

        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy, y_pred



#################stacking###################

import sklearn 

#New Feature
X_stacked = np.array([predictions, predictions_knn]).T

Stacking_tree = KNN_manhattan_Classifier(k=3)
Stacking_tree.fit(X_stacked, y_test)


print("開始預測")
stack_pred = Stacking_tree.predict(X_stacked)
print ("\nThe Val Accuracy of our new feature is:", sklearn.metrics.accuracy_score(y_test, stack_pred))

print ("\nThe Val Accuracy of our classifier old feature is:", sklearn.metrics.accuracy_score(y_test, predictions))
print ("\nThe Val Accuracy of our knn old feature is:", sklearn.metrics.accuracy_score(y_test, predictions_knn))










##############################################第二大題##################################################################

#重新分割成train, valiadation, test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_test,y_test,test_size=0.5,random_state=42,stratify=y_test)

# X_train.info()
# X_val.info
# X_test.info()
# y_train.info()
# y_test.info()


#複製原本dataframe，將布林值轉為int
boolean_columns = X_train.select_dtypes(include=['bool']).columns
for i in boolean_columns:
    X_train[i] = X_train[i].astype(int)
    X_val[i] = X_val[i].astype(int)
    X_test[i] = X_test[i].astype(int)
X_train_df = X_train.copy()
X_val_df = X_val.copy()
X_test_df = X_test.copy()


# X_train.info()
# X_val.info()
# X_test.info()
# y_train.info()
# y_val.info()
# y_test.info()


X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)



###############################  Randon Forest with 2-layer MLP  ###############################
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def train(self, X, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backpropagation
            loss = self.cross_entropy_loss(output, y)
            self.backward(X, y, output, learning_rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def cross_entropy_loss(self, output, y):
        m = y.shape[0]
        log_likelihood = -np.log(output[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, X, y, output, learning_rate):
        m = y.shape[0]

        # Output layer gradient
        dZ2 = output
        dZ2[range(m), y] -= 1
        dZ2 /= m

        # Backpropagation through the second layer
        dW2 = np.dot(self.a1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backpropagation through the first layer
        dZ1 = np.dot(dZ2, self.weights2.T)
        dZ1[self.z1 <= 0] = 0  # ReLU derivative
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.weights2 -= learning_rate * dW2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dW1
        self.bias1 -= learning_rate * db1


class DeepRandomForest:
    def __init__(self, num_trees, num_features, num_instances, num_classes):
        self.num_trees = num_trees
        self.num_features = num_features
        self.num_instances = num_instances
        self.num_classes = num_classes
        self.trees = []

    def train(self, X, y, num_epochs=100):
        for _ in range(self.num_trees):
            # Random subset of features and instances
            selected_features = np.random.choice(X.shape[1], self.num_features, replace=False)
            selected_instances = np.random.choice(X.shape[0], self.num_instances, replace=False)
            X_subset = X[selected_instances][:, selected_features]
            y_subset = y[selected_instances]

            # Train an MLP with the correct input size
            mlp = MLP(self.num_features, hidden_size=64, output_size=self.num_classes)
            mlp.train(X_subset, y_subset, epochs=num_epochs)

            # Add the trained MLP to the list of trees
            self.trees.append(mlp)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.num_classes))

        for tree in self.trees:
            # Use each tree to make predictions
            tree_output = tree.forward(X[:, :self.num_features])  # Use only selected features
            predictions += tree_output

        # Combine predictions using simple averaging
        final_predictions = predictions / self.num_trees

        return np.argmax(final_predictions, axis=1)
    
    
    
#宣告模型
deep_random_forest = DeepRandomForest(num_trees=5, num_features=10, num_instances=50, num_classes=2)

#訓練模型
deep_random_forest.train(X_train, y_train)

#驗證集表現
drf_y_val_pred = deep_random_forest.predict(X_val)
print ("\nThe Val Accuracy of our Randon Forest with 2-layer MLP is:", sklearn.metrics.accuracy_score(y_val, drf_y_val_pred))

#測試集表現
drf_y_test_pred = deep_random_forest.predict(X_test)
print ("\nThe Test Accuracy of our Randon Forest with 2-layer MLP is:", sklearn.metrics.accuracy_score(y_test, drf_y_test_pred))



