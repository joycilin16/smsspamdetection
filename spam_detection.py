#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


import pandas as pd


# In[3]:


import os


# In[4]:


df = pd.read_csv(r"D:\spam.csv", encoding='latin-1')


# In[5]:


df.head() 


# In[6]:


df.shape 


# In[7]:


df.v1.value_counts()


# In[8]:


df.isnull().sum() 


# In[9]:


df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df = df.rename(columns = {'v1':'label','v2':'message'}) 


# In[10]:


df.head() 


# In[11]:


df.shape


# In[12]:


df['label'] = df['label'].map({'ham': 0, 'spam': 1}) 


# In[13]:


df['length'] = df['message'].apply(len)
df.head() 


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns 


# In[15]:


plt.figure(figsize=(8,8))
g = sns.countplot(x='label', data=df)
p = plt.title('Countplot for Spam vs. Ham as balanced dataset', fontweight='bold')
p = plt.xlabel('Is SMS Spam?', fontweight='bold')
p = plt.ylabel('Count', fontweight='bold')  


# In[16]:


plt.rcParams['patch.force_edgecolor'] = True
plt.style.use('ggplot')
df.hist(column='length', by='label', bins=50, figsize=(11,5));


# In[17]:


for i in df.columns:
  print("Basic Statistics for Feature: {0}".format(i))
  print(df[i].describe())
  print("==========================================")


# In[18]:


import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[19]:


import string
def process_text(text):
  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)

  clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

  return clean_words 


# In[20]:


df['message'].head().apply(process_text)


# In[21]:


import pickle
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer=process_text)
message_bow = cv.fit_transform(df['message'])

# Creating a pickle file for the text transformation
pickle.dump(cv, open('cv-transform.pkl', 'wb'))


# In[22]:


message_bow.shape


# In[23]:


from sklearn.model_selection import train_test_split
X_train, x_validation, Y_train, y_validation = train_test_split(message_bow,
                                                                df['label'],
                                                                test_size=0.2,
                                                                stratify=df['label'],
                                                                random_state=0) 


# In[24]:


from sklearn import svm


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[26]:


from sklearn.model_selection import GridSearchCV, ShuffleSplit


def find_best_model(X, y):
    models = {
        'Logistic_Regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'Decision_Tree': {
            'model': DecisionTreeClassifier(splitter='best'),
            'parameters': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 10]
            }
        },
        
        'Random_Forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        
        'SVM': {
            'model': SVC(gamma='auto'),
            'parameters': {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }
        },

        'Multinomial_Naive_Bayes': {
            'model': MultinomialNB(),
            'parameters': {
                'alpha': [0.5, 1, 2, 3, 5, 10]
            }
        },

        'KNN': {
            'model': KNeighborsClassifier(),
            'parameters': {
                'n_neighbors': [1, 2, 3, 4, 5, 6, 8, 10, 20]
            }
        },

        'Extra_Tree': {
            'model': ExtraTreesClassifier(),
            'parameters': {
                'n_estimators': [1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 150, 200],
                'max_depth': [1, 2, 5, 6, 7, 8, 10, 20, 30],
                'criterion': ['gini', 'entropy']
            }
        }

    }

    scores = [] 
    cv_shuffle = ShuffleSplit(n_splits=7, test_size=0.20, random_state=0)
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])

find_best_model(X_train, Y_train)


# In[27]:


classifier = SVC(C=1.0, kernel='linear')
classifier.fit(X_train, Y_train)


# In[28]:


from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
y_pred = classifier.predict(X_train)
print(classification_report(Y_train, y_pred))
print()
print("Confusion Matrix: \n", confusion_matrix(Y_train, y_pred))
print()
print("Accuracy: ", accuracy_score(Y_train, y_pred))


# In[29]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# On validation data
y_pred = classifier.predict(x_validation)
print(classification_report(y_validation, y_pred))
print()
print("Confusion Matrix: \n", confusion_matrix(y_validation, y_pred))
print()
print("Accuracy: ", accuracy_score(y_validation, y_pred)) 


# In[30]:


classifier2 = MultinomialNB(alpha=3.0)
classifier2.fit(X_train, Y_train)
y_pred = classifier2.predict(X_train)
print(classification_report(Y_train, y_pred))
print()
print("Confusion Matrix: \n", confusion_matrix(Y_train, y_pred))
print()
print("Accuracy: ", accuracy_score(Y_train, y_pred))


# In[31]:


classifier = SVC(C=1.0, kernel='linear')
classifier.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
y_pred = classifier.predict(X_train)
print(classification_report(Y_train, y_pred))
print()
print("Confusion Matrix: \n", confusion_matrix(Y_train, y_pred))
print()
print("Accuracy: ", accuracy_score(Y_train, y_pred))


# In[33]:


classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
y_pred = classifier.predict(X_train)
print(classification_report(Y_train, y_pred))
print()
print("Confusion Matrix: \n", confusion_matrix(Y_train, y_pred))
print()
print("Accuracy: ", accuracy_score(Y_train, y_pred))


# In[34]:


import pickle

# Creating a pickle file for the classifier
filename = 'spam-detection-svm.pkl'
pickle.dump(classifier, open(filename, 'wb')) 


# In[35]:


pip install prettytable


# In[36]:


from prettytable import PrettyTable

x = PrettyTable()
x.field_names = ["Model Used", "Accuracy"]

x.add_row(["Logistic Regression", 0.9729])
x.add_row(["Decision Tree Classifier", 0.9481])
x.add_row(["Random Forest Classifier", 0.9621])
x.add_row(["Support Vector Machine", 0.9757])
x.add_row(["Multinomial Naive Bayes", 0.9725])
x.add_row(["KnearestNeighbor", 0.9336])
x.add_row(["Extra Tree Classifier", 0.9369])

print(x)


# In[37]:


X_Validation =("Go until jurong point, crazy.. Available only ...")


# In[38]:


print(y_validation)


# In[39]:


X_Validation =("Free entry in 2 a wkly comp to win FA Cup fina...	")


# In[40]:


print(y_validation)


# In[41]:


x_validation


# In[42]:


y_validation


# In[ ]:




