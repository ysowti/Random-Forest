
# coding: utf-8

# # Detect Cardiologists Using Random Forest (Solution 2)
# 
# In this solution I use Random Forest model to classify cardiologists from others in a group of physicians with unknown specialty. For this reason I use 'procedure_code' as the main feature to train random forest classifier (RFC). In this work I also did a more thorough evaluation of the tuned model based on cross validation, precision recall tradeoff and ROC curve. The procedure is divided into 6 steps including preprocessing and feature extraction, modeling, precision/recall tradeoff, test set evaluation and prediction on 'unknown'.

# In[1]:

# First of all I import all the libraries that I need during analysis
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

import sklearn
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer

get_ipython().magic('matplotlib inline')


# # Preprocessing and Feature Extraction
# 
# In the first step I preprocess the data to extract feature vectors and target vectors that later on can be used to train the model. First of all, physicians and procedure datasets are imported into 'phys' and 'proc' dataframes, respectively. Next, all distinct 'procedure_code's are extracted to create a transformer called LabelBinarizer. LabelBinarizer converts categorical data like 'procedure_code' to integer categories and from integer categories to one-hot vectors (vectors with attribute equal to 1 for one category and 0 otherwise). at next step I split physicians with known 'specialty' from unknown ones using unknown indexes. The dataset with known specialty is called 'phys_known' and the other one 'phys_unknown'.
# 
# After splitting the data, I use 'feature_extract' function to form a feature matrix where all 'procedure_code's available for each phys['id'] are expressed as a vector taking binary values on each procedure, and then all these vectors are sumed up vertically to form a feature vector per phys['id']. This function is applied on 'phys_unknown' to extract feature vectors for physicians with known 'specialty'. Then, I use 'extract_target' function to extract traget values used to form the target vector 'y'.
# 
# Now that we have the required features and target values to train classifer, we need to split the data into train and test sets where train data will be used to train the model. For this instead of using the common approach of random sampling I use stratified sampling that divides the population into homogeneous subgroups called starta, and the irght number of instances is sampled from each stratum to guarantee that the test set is representative of the overall population. Train data input and output is called X_train and y_train and for the test data it is called X_test and y_test. Test_data later on will be used to evaluate performance of the classifier.

# In[2]:

# import physicians data
phys = pd.read_csv('../physicians.csv')


proc = pd.read_csv('../procedures.csv')


# Get all the distinct 'procedure_code's and use them to create LabelBinarizer 
proc_code_list = sorted(proc['procedure_code'].unique())
proc_label_binarizer = LabelBinarizer(sparse_output=False)
proc_label_binarizer.fit(proc_code_list)


# In[3]:

# split phsician data into unknown and known

# Get unknown indexes in physicians and split physicians based on that
_choose = ['Unknown' in specialty for specialty in phys['specialty']]
phys_unknown = phys[_choose]
phys_known = phys[[not ii for ii in _choose]]


# In[4]:

# This function forms the feature matrix
# Features are all procedure_code expressed in a vector taking binary values on each procedure (True/False)
def extract_features(phys):
    X = []
    for nrow, physician_id in enumerate(phys['id']):
        # get all the 'procedures_code' conduced by a physician
        _proc = proc[proc['physician_id'] == physician_id]['procedure_code']
        # binarizing all available 'procedure_code' using label_binarizer
        _bin = proc_label_binarizer.transform(_proc.values)
        # Take vertical sum to form a feature vector and append to X
        X.append(_bin.sum(axis=0))
    
    return np.asarray(X)
# Apply feature extraction function to extract feature for physicians with known 'specialty'
X = extract_features(phys_known)


# In[5]:

# This function extract target values that are used to form the target vector (y)
def extract_targets(phys):
    return (phys['specialty'] == 'Cardiology').values

y = extract_targets(phys_known)


# In[6]:

# Stratified shuffle-split method is used to split the data into train and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X, y):
    X_train = X[train_index,:]
    y_train = y[train_index]
    
    X_test = X[test_index,:]
    y_test = y[test_index]


# # Modeling
# 
# At this step I use the generated training dataset to train a random forest model and use Scikit-Learn’s cross-validation feature to evaluate it. First of all, I creat an RFC object and then fit the model to the training data X_train and y_train obtained from last step. I run cross vaidation of 5-fold and get the average accuracy score for each fold. As can be seen all the accuracy scores are close to 0.97. As the next step, I use cross_val_predict() to performs 5-fold cross-validation and to return the predictions made on each test fold. This means that I get a clean prediction for each instance in the training set (“clean” meaning that the prediction is made by a model that never saw the data during training). After getting these predictions, the last step is to compare them with actual values and get the confusion matrix, precision, recall and f1 scores.
# 
# Confusion matrix indcate that 113117 instances of training data are identified correctly as class 0 (non-cardiologists) that are true negatives, and 6274 cases are correctly classified to 1 (specialty of cardiology) as true positives. On the other hand, there are 345 false negatives and 207 false positives. It can also be seen that obtained precision is approximately 0.97, and recall is roughly 0.95 giving us f1-score of about 0.96.

# In[7]:

# train random forest classifier on training data
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)


# In[8]:

# get cross validation score for 50fold cross validation
cross_val_score(forest_clf, X_train, y_train, cv=5, scoring="accuracy")


# In[9]:

# Run cross vaidation of 5-fold and get clean prediction on each fold
y_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=5)

# calculate confusion matrix and calculate precision, recall and f1 score

print("Confusion Matrix: \n", confusion_matrix(y_train, y_train_pred))
print("Precision Score: ", precision_score(y_train, y_train_pred))
print("Recall Score: ", recall_score(y_train, y_train_pred))
print("F1 Score: ", f1_score(y_train, y_train_pred))


# # Presicion/Recall Tradeoff
# 
# At this step We study the precision/recall tradeoff for the trained classifier in last step. First of all, I use corss_val_predict, but this time instead of returning prediction I ask it to return prbabilities of each instance per class. the Random Forest Classifier has a predict_proba() method that returns an array containing a row per instance and a column per class, each containing the probability that the given instance belongs to the given class. Then, it uses threshold to decide which instance belongs to which class. Since here exists only two classes, by default this threshold is 0.5. Next, I plot precision and recall when this threshold varies from 0 to 1. You can see the tradeoff between precision and recall, as with very high precision in return you get a low recall and vice versa. Also, we can see that 0.5 seems to be a good threshold because this is the threshold that yields to a high value for precision and recall.

# In[24]:

# Precision/Recall tradeoff

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=5, method='predict_proba')

y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_forest)


# In[26]:

# Plot precision and recal vs threshold
fig = plt.figure(figsize=(15,7))
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(thresholds, precisions[:-1], label = 'precision')
axes.plot(thresholds, recalls[:-1], label = 'Recall')
axes.set_xlabel('Threshold')
axes.set_title('Precision and recall versus the decision threshold')
axes.legend()


# # The Receiver Operating Characteristic Curve (ROC)
# 
# As another performance evaluation here I use receiver operating characteristics curve (ROC) that plots true positive rate against the false positive rate. Once again there is a tradeoff: the higher the recall (TPR), the more false positives
# (FPR) the classifier produces. The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner). One way to compare classifiers is to measure the area under the curve (AUC). A perfect classifier will have a ROC AUC equal to 1. Here AUC that trained cassifier obtains is 0.985 that pretty close 1 that is desirable.

# In[27]:

# The ROC curve
fpr, tpr, thresholds = roc_curve(y_train, y_scores_forest)


# In[28]:

# plot ROC curve
plt.figure(figsize=(15,7))
plt.plot(fpr, tpr)
plt.plot([0, 1],[0, 1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')


# In[14]:

# Compute AUC for the classifier
roc_auc_score(y_train, y_scores_forest)


# # Test Set Evaluation
# 
# In this step I use test data to evaluate the RFC that was created in modeling step. Fist of all, I use the tuned RFC to get predictions for test data and then use these predictions to obtain confusion matrix and precision, recall and f1 scores similar to modeling step with a difference that here completely outside data is used to get these scores. As can be seen, precision 0f 0.966 is a bit higher than recall of about 0.946 resulting in f1 score of 0.956. That is because false negative is lager that false positve meaning that there are more instances of class 1 being falsely classified as 0 than cases of class 0 being falsely identified as class 1.

# In[15]:

# Run predictions on test data using modeled RFC
y_test_pred = forest_clf.predict(X_test)


# In[16]:

# calculate confusion matrix and calculate precision, recall and f1 score
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score
print("Confusion Matrix: \n", confusion_matrix(y_test, y_test_pred))
print("Precision Score: ", precision_score(y_test, y_test_pred))
print("Recall Score: ", recall_score(y_test, y_test_pred))
print("F1 Score: ", f1_score(y_test, y_test_pred))


# # Prediction on the 'Unkown'
# 
# As the last step I use the random forest model that I trained and evaluated in last steps to run predictions in instances of physicians with unknown specialty. It results in 3631 physicians being detected as cardilogists out of 11096.

# In[17]:

# Extract features for phys_unknwon
X_ = extract_features(phys_unknown)


# In[18]:

# Get predictions for phys_unknown using RFC and get the total number of cardiologists
y_unknown_pred = forest_clf.predict(X_)
predicted_counts = y_unknown_pred.sum()
total_predicted = len(y_unknown_pred)
print("Number of carodiologists found is {} out of {}".format(predicted_counts,total_predicted))

print("Percentage of carodiologists found out of total is: ", (y_unknown_pred.sum()/len(y_unknown_pred)))

