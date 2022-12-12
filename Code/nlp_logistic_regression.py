### --- Logistic regression and SHAP ---###

# import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import shap
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# read the file
df = pd.read_csv('df_short.csv')
print(df.info())
print(df.shape)


# recode label class: sentiment (recommend = 1, not recommend = 0)
df['label'] = df['reviews.doRecommend'].apply(lambda x: '1' if x == True else '0')
print(df['label'].value_counts())


df = df[['label','clean_text']]
print(df.head())


# ------ vectorization: TFDIF --------
doc_complete = list(df.clean_text)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(doc_complete)
terms = tfidf.get_feature_names()
#print("Terms:",terms)
print("data size after vectorization:", X.shape)


# --------- Split into train and test set -------------
# define X and Y and target labels
Y = df.values[:, 0]
X = X

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


print(len(y_train))
print(X_train.shape)

print(X_test.shape)
print(y_test.shape)


# ----- Logistic regression -------
clf_lr = LogisticRegression().fit(X_train, y_train)
predicted_lr = clf_lr.predict(X_test)

print('Results from Logistic Regression:')
print(metrics.classification_report(y_test, predicted_lr, zero_division=True))
print(metrics.confusion_matrix(y_test, predicted_lr))
target_names=['NotRecommend','Recommend']

# confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_lr)
df_cm = pd.DataFrame(conf_matrix, index=target_names, columns=target_names)
#df_cm = pd.DataFrame(conf_matrix)
plt.figure(figsize=(5,5))
hm_ab = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm_ab.yaxis.set_ticklabels(hm_ab.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm_ab.xaxis.set_ticklabels(hm_ab.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=15)
plt.xlabel('Predicted label',fontsize=15)
plt.tight_layout()
plt.show()


# SHAP
explainer = shap.LinearExplainer(clf_lr, X)
shap_values= explainer.shap_values(X)
X_array = X.toarray()

shap.summary_plot(shap_values, X_array,feature_names=terms)