import pandas as pd
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score

df = pd.read_excel('training.xlsx',header=None)
df.columns=['polarity','text']
df.text.str.replace('[^\x00-\x7F]+', "")    #Removing all non-ASCII characters
df.text.str.replace('(\\n)',"")

y = df['polarity']
x = df.drop('polarity', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

CV=CountVectorizer(stop_words=ENGLISH_STOP_WORDS)  #Count vectorizer is used over tfidf because it is giving better accuracy and the
                                                   #accuracy is also being verified by doing cross validation. This might be happening bcoz
                                                   #in the given training dataset the commom words only are classifying the text into +ve
                                                   #and -ve. tf-idf might be reducing their importance in contributing to polarity due to
                                                   #more inverse document frequency.

#tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,lowercase=True,ngram_range=(1,3),max_df=0.9, min_df=0.1)  #only 17 words were considered
                                                                                                                 # in the vector bcoz of selection
                                                                                                                 #ranges given as parameters
                                                                                                                 #in tfidf function.

x_train=pd.DataFrame(CV.fit_transform(x_train['text'].values.astype('U')).toarray().tolist())
x_test=pd.DataFrame(CV.transform(x_test['text'].values.astype('U')).toarray().tolist())

# Applying grid search on svm and setting the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']      #precision relates to false positive while recall relates t false negative

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,scoring='%s_macro' % score)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    print("Detailed classification report for SVM:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_pred = clf.predict(x_test)
    print(metrics.classification_report(y_test, y_pred))
    print('SVM: Area under the ROC curve:',metrics.roc_auc_score(y_test, y_pred))

scores = cross_val_score(clf, x_train, y_train, cv=6)  #to check if results of count vectorizer are correct for all sections of train data

#applying grid search on decision tree
param_grid = {'max_depth': np.arange(3, 10)}
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)
print("Detailed classification report for Decision tree:")
print(metrics.classification_report(y_test, y_pred))
print ('DecisionTree: Area under the ROC curve = ',metrics.roc_auc_score(y_test, y_pred))

data = pd.read_excel('testdata.xlsx',header=None)
data=data[0]
data=pd.DataFrame(tfidf.transform(data.astype('U')).toarray().tolist())
result=tree.predict(data)
output=pd.DataFrame()
output['']=result
output.to_csv('submit.csv')