#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus


# ### Data

# In[2]:


data = pd.read_excel("D:/Semester 6/Data Mining/Tugas Klasifikasi/heart.xlsx")
data


# In[18]:


data.info()


# ### <center>A. *PREPROCESSING*</center>

# ### 1. Merubah Nama dan Tipe Variabel
# Agar variabel terdeteksi sesuai dengan tipenya, maka dilakukan perubahan tipe variabel. Selain itu, nama variabel diubah dengan nama yang lebih detail.

# In[3]:


data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 
                'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope',
                'num_major_vessels', 'thalassemia', 'target']


# In[10]:


data.dtypes


# Tipe data beberapa variabel masih tidak tepat, seperti sex, cp, dan sebagainya seharusnya merupakan tipe kategorik (objek)

# In[45]:


data['sex'] = data['sex'].astype('object')
data['chest_pain_type'] = data['chest_pain_type'].astype('object')
data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype('object')
data['rest_ecg'] = data['rest_ecg'].astype('object')
data['exercise_induced_angina'] = data['exercise_induced_angina'].astype('object')
data['st_slope'] = data['st_slope'].astype('object')
data['thalassemia'] = data['thalassemia'].astype('object')
data['target'] = data['target'].astype('object')


# ### 2. Deteksi *Missing Value*

# In[21]:


np.sum(data.isnull())


# Tidak ada variabel yang memiliki *missing value* pada data ini

# ### 3. Deteksi *Outlier*
# Deteksi *outlier* akan dilakukan dengan menggunakan *boxplot* pada data bertipe numerik

# In[13]:


data.dtypes


# Terdapat 6 variabel bertipe numerik. Sehingga deteksi outlier dilakukan pada 6 variabel tersebut

# In[23]:


plt.figure(figsize = (11,8))
plt.subplot(231)
sns.boxplot(x = data['age'], saturation = 1, width = 0.8, color = 'seagreen', orient = 'v')
plt.ylabel('age', fontsize = 14)
plt.subplots_adjust(wspace = 0.6)
plt.subplot(232)
sns.boxplot(x = data['resting_blood_pressure'], saturation = 1, width = 0.8, color = 'seagreen', orient = 'v')
plt.ylabel('resting_blood_pressure', fontsize = 14)
plt.subplot(233)
sns.boxplot(x = data['cholesterol'], saturation = 1, width = 0.8, color = 'seagreen', orient = 'v')
plt.ylabel('cholesterol', fontsize = 14)
plt.subplot(234)
sns.boxplot(x = data['max_heart_rate_achieved'], saturation = 1, width = 0.8, color = 'seagreen', orient = 'v')
plt.ylabel('max_heart_rate_achieved', fontsize = 14)
plt.subplot(235)
sns.boxplot(x = data['st_depression'], saturation = 1, width = 0.8, color = 'seagreen', orient = 'v')
plt.ylabel('st_depression', fontsize = 14)
plt.subplot(236)
sns.boxplot(x = data['num_major_vessels'], saturation = 1, width = 0.8, color = 'seagreen', orient = 'v')
plt.ylabel('num_major_vessels', fontsize = 14)
plt.show()


# ### 4. Transformasi Data

# Transformasi data dibutuhkan pada saat klasifikasi, namun untuk visualisasi masih diperlukan data yang tidak ditransformasi,
# sehingga dibuat varibel baru yaitu dataset yang berisikan data yang telah dilakukan tranformasi. **Transformasi dilakukan setelah ada variabel dummy di bawah**

# ### <center>B. *Summary Statistics*</center>

# In[9]:


list = ['sex','chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope','target','thalassemia']
num = data.drop(list,axis = 1 )
num.describe()


# ### <center>C. Visualisasi Data</center>

# ### Jumlah yang Terkena Penyakit Jantung dan Tidak

# In[24]:


data.target.value_counts().plot(kind="bar", color=["seagreen", "palegreen"])
x = np.arange(2)
plt.xticks(x, ('Sakit Jantung', 'Tidak Sakit Jantung'))


# Berdasarkan plot di atas, terdapat 526 pasien terkena penyakit jantung, sementara 499 lainnya tidak terkena penyakit jantung. Variabel ini yang akan dijadikan kelas yang akan prediksi dalam klasifikasi. Kedua kategori mempunya frekuensi yang tidak berbeda jauh atau selisihnya kecil. Sehingga, dalam klasifikasi digunakan akurasi saja sudah cukup untuk mengukur kebaikan dari metode klasifikasi

# ### Korelasi Antar Variabel

# #### 1. Heatmap

# In[52]:


corr_matrix = data.corr()
fig, ax = plt.subplots(figsize=(8, 8))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="BuGn",
                 annot_kws={"size": 13});
bottom, top = ax.get_ylim()
sns.set(font_scale=2)
ax.set_ylim(bottom + 0.5, top - 0.5)


# Heatmap di atas menunjukkan besarnya korelasi Pearson. Korelasi Pearson hanya dapat digunakan pada data numerik. Ini merupakan salah satu manfaat dari perubahan type variabel yang sesuai. Jika tipe variabel tidak dirubah, maka data kategori seperti jenis kelamin (sex), thalassemia, dsb akan ikut dihitung korelasi pearsonnya.

# #### 2. *Pair Plot*

# In[62]:


# Tidak jadi dipakai
num_var = ['age','resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
sns.pairplot(data[num_var], kind='scatter', diag_kind='hist',palette = "seagreen")
sns.set(font_scale=0.5)
plt.show()


# ### Korelasi Masing-Masing Variabel dengan Target

# In[67]:


df = pd.read_excel("D:/Semester 6/Data Mining/Tugas Klasifikasi/heart.xlsx")
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 
                'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope',
                'num_major_vessels', 'thalassemia', 'target']
sns.set(font_scale=1.5)
df.drop('target', axis=1).corrwith(df.target).plot(kind='bar', color = 'green', grid=True, figsize=(12, 8))


# ### Perbandingan Jumlah Orang Yang Terkena Penyakit Jantung dan Tidak Berdasarkan Variabel Kategori

# In[4]:


categorical_val = []
continous_val = []
for column in data.columns:
    print('==============================')
    print(f"{column} : {data[column].unique()}")
    if len(data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[20]:


categorical_val


# In[72]:


plt.figure(figsize=(15, 15))

for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    data[data["target"] == 0][column].hist(bins=35, color='red', label='Penyekit Jantung = Tidak', alpha=0.6)
    data[data["target"] == 1][column].hist(bins=35, color='darkgreen', label='Penyakit Jantung = Ya', alpha=0.6)
    plt.legend()
    plt.xlabel(column, fontsize = 13)


# ### Perbandingan Jumlah Orang Yang Terkena Penyakit Jantung dan Tidak Berdasarkan Variabel Numerik

# In[74]:


plt.figure(figsize=(15, 15))

for i, column in enumerate(continous_val, 1):
    plt.subplot(3, 2, i)
    data[data["target"] == 0][column].hist(bins=35, color='darkgoldenrod', label='Penyakit Jantung = Tidak', alpha=0.6)
    data[data["target"] == 1][column].hist(bins=35, color='darkgreen', label='Penyakit Jantung = Ya', alpha=0.6)
    plt.legend()
    plt.xlabel(column, fontsize = 13)


# ### <center>D. Klasifikasi</center>

# ### Variabel Dummy

# In[83]:


categorical_val.remove('target')
dataset = pd.get_dummies(data, columns = categorical_val)
dataset


# In[84]:


# Run Variabel Dummy
s_sc = StandardScaler()
col_to_scale = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
dataset.head()


# In[8]:


print(data.columns)
print(dataset.columns)


# In[85]:


categorical_val = []
continous_val = []
for column in data.columns:
    print('==============================')
    print(f"{column} : {data[column].unique()}")
    if len(data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[10]:


categorical_val


# In[86]:


categorical_val.remove('target')
dataset = pd.get_dummies(data, columns = categorical_val)
dataset.head()


# ### Data Training & Testing dengan Machine Learning

# In[87]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[ ]:





# ### Cross Validation

# In[88]:


X = dataset.drop('target', axis=1)
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ### 1. *Support Vector Machine*

# #### *Parameter Tunning*

# In[89]:


svm_model = SVC(kernel='rbf', gamma=0.1, C=1.0)

params = {"C":(0.1, 0.5, 1, 2, 5, 10, 20), 
          "gamma":(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1), 
          "kernel":('linear', 'poly', 'rbf')}

svm_grid = GridSearchCV(svm_model, params, n_jobs=-1, cv=5, verbose=1, scoring="accuracy")
svm_grid


# In[ ]:


svm_grid.fit(X_train, y_train) ## Maaf mas ke run lagi dan lama.


# In[17]:


svm_grid.best_estimator_


# #### Akurasi

# In[18]:


svm_model = SVC(C=2, gamma=0.5, kernel='rbf')
svm_model.fit(X_train, y_train)

print_score(svm_model, X_train, y_train, X_test, y_test, train=True)
print_score(svm_model, X_train, y_train, X_test, y_test, train=False)


# In[19]:


test_score = accuracy_score(y_test, svm_model.predict(X_test)) * 100
train_score = accuracy_score(y_train, svm_model.predict(X_train)) * 100
tuning_results_data = pd.DataFrame(data=[["Tuned SVM", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
tuning_results_data


# #### *Heatmap Confusion Matrix*

# In[60]:


pred = svm_model.predict(X_test)
sv_cm=confusion_matrix(y_test, pred)
ax = sns.heatmap(sv_cm,annot=True,cmap="Blues",fmt="d",cbar=False, linewidths=0.5,
                 xticklabels=["Haven't Disease", "Have Disease"], 
                 yticklabels=["Haven't Disease", "Have Disease"],
                 annot_kws={"size": 13});
plt.xlabel('Predicted Values')
plt.ylabel('True Values');
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# ### 2. *Decision Tree*

# #### *Parameter Tunning*

# In[22]:


params = {"criterion":("gini", "entropy"), 
          "splitter":("best", "random"), 
          "max_depth":(list(range(1, 20))), 
          "min_samples_split":[2, 3, 4], 
          "min_samples_leaf":list(range(1, 20))
          }

tree = DecisionTreeClassifier(random_state=42)
grid_search_cv = GridSearchCV(tree, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3, iid=True)


# In[23]:


grid_search_cv.fit(X_train, y_train)


# In[24]:


grid_search_cv.best_estimator_


# #### Akurasi

# In[25]:


tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=10,
                              min_samples_leaf=1, 
                              min_samples_split=2, 
                              splitter='random')
tree.fit(X_train, y_train)

print_score(tree, X_train, y_train, X_test, y_test, train=True)
print_score(tree, X_train, y_train, X_test, y_test, train=False)


# In[26]:


test_score = accuracy_score(y_test, tree.predict(X_test)) * 100
train_score = accuracy_score(y_train, tree.predict(X_train)) * 100

results_data_2 = pd.DataFrame(data=[["Tuned Decision Tree Classifier", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
tuning_results_data = tuning_results_data.append(results_data_2, ignore_index=True)
tuning_results_data


# In[45]:


dot_data = export_graphviz(tree, out_file=None, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# #### *Heatmap Confusion Matrix*

# In[37]:


pred = tree.predict(X_test)
tr_cm=confusion_matrix(y_test, pred)
ax = sns.heatmap(tr_cm,annot=True,cmap="Blues",fmt="d",cbar=False, linewidths=0.5,
                 xticklabels=["Haven't Disease", "Have Disease"], 
                 yticklabels=["Haven't Disease", "Have Disease"]);
plt.xlabel('Predicted Values')
plt.ylabel('True Values');
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# ### 3. *Naive Bayes*

# #### Akurasi

# In[38]:


nb = GaussianNB()
nb.fit(X_train, y_train)


# In[39]:


## Akurasi Training
nb.score(X_train, y_train)


# In[40]:


## Akurasi Testing
nb.score(X_test, y_test)


# #### *Heatmap Training*

# In[42]:


pred = nb.predict(X_train)
nb_cm=confusion_matrix(y_train, pred)
ax = sns.heatmap(nb_cm,annot=True,cmap="Blues",fmt="d",cbar=False, linewidths=0.5,
                 xticklabels=["Haven't Disease", "Have Disease"], 
                 yticklabels=["Haven't Disease", "Have Disease"]);
plt.xlabel('Predicted Values')
plt.ylabel('True Values');
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# #### *Heatmap Testing*

# In[41]:


pred = nb.predict(X_test)
nb_cm=confusion_matrix(y_test, pred)
ax = sns.heatmap(nb_cm,annot=True,cmap="Blues",fmt="d",cbar=False, linewidths=0.5,
                 xticklabels=["Haven't Disease", "Have Disease"], 
                 yticklabels=["Haven't Disease", "Have Disease"]);
plt.xlabel('Predicted Values')
plt.ylabel('True Values');
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

