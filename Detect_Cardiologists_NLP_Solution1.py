
# coding: utf-8

# # Detect Cardiologists using NLP (Solution 1)
# 
# In this Solution to the problem I use Natural Language Processing (NLP) to detect which physicians with unknown specialty are highly likely to be cardilogists. For this I divide all procedure into five steps, explanatory data analysis, preprocessing and feature extraction, modeling, evaluation and application. The main feature that we are going to use in this approach to train the classifier is 'procedure' colum, and the classifer that is going to be used is Multinomial Naive Bayesian.

# # Explanatory Data Analysis
# 
# In the first step which is explanatory data analysis, it is discovered that there are overall 71 unique specialties, out of which the ones with specialty as unknown form the largest group with the the highest number of physicians and the ones with specialty as cardiology form the second largest group. In this step we also see that the databses 'physician' and 'procedure' have the same number of unique identifiers, and therefore we can merge these two dataframes to simplify our analysis.

# In[1]:

# First we import required libraries for data processing
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# Import physician database as dataframe from CSV file
physicians = pd.read_csv('physicians.csv')
physicians.head()


# In[3]:

# Get general information about physicians dataframe
physicians.info()


# In[4]:

# How many unique values are there in specialty column
physicians['specialty'].nunique()


# In[5]:

# Get value counts for values appearing in specialty column
physicians['specialty'].value_counts().head(10)


# In[6]:

# Histogram for each specialty
plt.figure(figsize=(17, 5))
plt.ylabel("Counts of specialty")
physicians['specialty'].value_counts().plot(kind='bar')


# In[6]:

# Import procedures database as dataframe from CSV file
procedures = pd.read_csv('procedures.csv')
procedures.head()


# In[7]:

# Get general information about physicians dataframe
procedures.info()


# In[9]:

# Number of unique identifiers in procedures dataframe
procedures['physician_id'].nunique()


# In[10]:

# Number of unique identifiers in physicians dataframe
physicians['id'].nunique()


# # Preprocessing and Feature Extraction
# 
# In the second step which is preprocessing and feature extraction, the physicians dataframe is merged with the procedures dataframe to create a combined table called 'physproc'. Since in this solution I use 'procedure' column as the main feature for the classifier, I only extract 'specialty' and 'procedure' columns from the 'physproc' table and group it by the 'physician_id' through concatenating all the texts in 'procedure' for each physician. The resulting dataframe is called 'physproc_grouped'. Next, I split 'physproc_grouped' into two sets of data, first dataset with known 'specialty', called 'known_physproc', that will be used to train the classifier, and the second dataset with unknown 'specialty', called 'unknown_physproc', on which the classifier will be applied to detect cardiologists. Finally, a new column, called 'Cardiology' is added to the 'known_physproc' dataset where it is labeled as '1' for specialty as 'cardiology' and '0' for others. Now, we have feature 'procedure' ready to use with label in order to train our model in next step.

# In[11]:

# Merge physicians and procedures dataframe on identifiers
physproc = pd.merge(left=physicians,right=procedures,how='inner',left_on=['id'],right_on=['physician_id'])
physproc.head()


# In[12]:

"""
extract 'specialty' and 'procedure' columns from the 'physproc' table and group it by the 
'physician_id' through concatenating all the texts in 'procedure' for each physician

""" 
physproc_grouped = physproc[['physician_id','specialty','procedure']].groupby(['physician_id','specialty'])['procedure'].apply(lambda x: '_'.join(x)).reset_index()
physproc_grouped.head()


# In[13]:

# A sample procedure feature for the first row
physproc_grouped['procedure'][0]


# In[14]:

"""
Split the physproc_grouped into known_physproc with known specialties
and unknown_physproc with unknown specialties

"""
known_physproc = physproc_grouped[physproc_grouped['specialty'] != 'Unknown']
known_physproc.head()


# In[15]:

""" 

A new column, called 'Cardiology' is added to the 'known_physproc' dataset 
where it is labeled as '1' for specialty as 'cardiology' and '0' for others.

"""

known_physproc['Cardiology'] = known_physproc['specialty'].apply(lambda s : 1 if s == 'Cardiology' else 0)


# In[16]:

known_physproc.head()


# In[17]:

unknown_physproc = physproc_grouped[physproc_grouped['specialty'] == 'Unknown']
unknown_physproc.head()


# In[18]:

unknown_physproc.info()


# # Modeling
# 
# In the third step I create a pipleline of transformers that is applied on the text data from procedure feature and the labels from the 'known_physproc' formed in previouse step. This pipeline which is called 'cardio_detector' is formed of two transformers, one is CountVectorizer called 'bow' (Bag of Words), and the second one which is multiomial Naive Bayesian model called 'classifier'. 
# 
# However, the text data in 'procedure' column is formed of a group of words seperated by underscore that also includes stop words in english that are very common words like ('the', 'a', etc..) that add no value to our analysis. For that reason I define a function called 'text_process' that first of all splits the string data on each feature in each row corresponding to a physician_id into a list of Words (called 'ListOfWords') and then removes all stopwords from it. To have the list of stopwords I use NLTK library for NLP and import stopwords from it. 'text_process' will act as analyzer in CountVectorizer.
# 
# 'bow' transformer uses the function 'text_process' as its analyzer to convert these strings in 'procedure' column into lists of tokens (also known as lemmas) and then transforms it into a vector that machine learning models can understand. This will convert a collection of text documents to a matrix of token counts. The next transformer is 'classifier' based on Naive Bayesian classifier that is trained to classifiy dataset into two groups of label '1' for 'cardiologists' and '0' for others. The reason I choose Multinomial NB model is that it usually works well with text data and people usually use it for applications like spam detection.
# 
# After creating pipeline of transformers, the data is split into train and test sets using train_test_split method from sklearn library with test size of 30% of all the data. In other words, the 'procedure' column, as the input feature, is splitted into two sets of procedure_train as train set and procedure_test as test set, and the output label from column 'Cardiology' is splitted into two sets of label_train and label_test. The train set is used for training the model and test set will be used in next step to evaluate the model. Finally, the 'cardio_detector' pipeline is trained on the train data (procedure_train as input feature and label_train as output) and is fit creating a pipeline object that can be used for prediction.

# In[19]:

# import stopwords method from nltk library
from nltk.corpus import stopwords


# In[20]:

# text process in a function that converts a string into a list of words without english stopwords in it
def text_process(mess):
    ListOfWords = mess.split('_')
    return [word for word in ListOfWords if word.lower() not in stopwords.words('english')]


# In[21]:

# import SciKit Learn's CountVectorizer.
from sklearn.feature_extraction.text import CountVectorizer


# In[22]:

# import Multinomial Naive Baysian classifier from sklearn library
from sklearn.naive_bayes import MultinomialNB


# In[23]:

# Create a pipeline of transformers including string vectorizer and classifier
from sklearn.pipeline import Pipeline

cardio_detector = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('classifier', MultinomialNB()),  # Naive Bayes classifier
])


# In[24]:

# Split train test data using train_test_split method from sklearn
from sklearn.model_selection import train_test_split

procedure_train, procedure_test, label_train, label_test = \     # Use test size of 30% in train test split
train_test_split(known_physproc['procedure'], known_physproc['Cardiology'], test_size=0.3)


# In[25]:

# train the pipeline model of cardio detector on training dataset
cardio_detector.fit(procedure_train,label_train)


# # Model Evaluation
# 
# In the fourth step I use the test set that was created using train_test_split in the previouse step to evaluate performance of the model. For that I use the pipeline object called 'cardio_detector' that was fit to the training data in the previouse step, and apply it to the input feature of the test set as 'procedure_test' to generate predictions as test_predictions. Next, these prediction are compared with the available labels for test dataset to evaluate accuracy of the model. I use SciKit Learn's built-in classification report, which returns precision, recall, f1-score, and a column for support (meaning how many cases supported that classification).
# As can be seen from Evaluation report the model resulted in on average 95% accuracy (precision, recall and f1 score of 95%) for both label '1' and label '0' outputs. For label '1' outputs that correspond to the cases that are predicted as 'cardiology', precision is 0.9 meaning that 90% of positive predictions as '1' are actually '1', and recall is 0.96 meaning that model has been able to predct 96% of actual 1's as '1'. As a result, we can conclude that model is predcitiing more cardiologists than the ones actually exist.

# In[26]:

# Use cardio_detector model to generate predictions based on test features 'procedure_test' 
test_predictions = cardio_detector.predict(procedure_test)
print(test_predictions)


# In[27]:

# I use classification_report from sklearn to return precision, recall and f1 score obtained on the test set
from sklearn.metrics import classification_report
print (classification_report(label_test, test_predictions))


# # Model Application
# 
# As the final step I apply the trained model on the data set with unknown specialties called 'unknown_physproc' and replace the results with the 'specialty' column in this dataset. The result shows that out of all 11092 physicians in this set, 3953 are detected as cardiologists.

# In[28]:

# run model to creat prediction on the dataset with unknonw specialty
application = unknown_physproc['procedure']
unknown_predictions = cardio_detector.predict(application)


# In[29]:

total = len(unknown_predictions)
new_cardios = unknown_predictions.sum()


# In[30]:

print("Number of carodiologists found is {} out of {}".format(new_cardios,total))


# In[31]:

unknown_physproc['specialty'] = ['cardiology' if s == 1 else 'others' for s in unknown_predictions]


# In[33]:

# The unknown dataset is no longer unknown with added predictions as its specialty column
unknown_physproc.head()

