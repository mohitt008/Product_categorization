import numpy as nm
import pandas as pd
import sklearn.cross_validation
import sklearn.feature_extraction.text
import sklearn.metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
#import sklearn.neural_network.MLPClassifier

import csv

names = ['product_name','category','sub_category']
data = pd.read_csv("classification_training_data.csv")
holdup = pd.read_csv("classification_testing_data.csv")

train,test = sklearn.cross_validation.train_test_split(data, train_size= .7)
train_data, test_data = pd.DataFrame(train, columns= names), pd.DataFrame(test, columns = names)
holdup_data = holdup.product_name
vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words= 'english')
#vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words= 'english', ngram_range = (1,2))

#vectorizer = sklearn.feature_extraction.text.TfidfTransformer()

train_matrix = vectorizer.fit_transform(train_data['product_name'])
test_matrix = vectorizer.transform(test_data['product_name'])
holdup_matrix = vectorizer.transform(holdup_data)
#print test_matrix
cat1 = []
sub_cat1 = []
for rows in data.category:
	cat1.append(rows)
for rows in data.sub_category:
	sub_cat1.append(rows)
cat = list(set(cat1))
sub_cat = list(set(sub_cat1))


train_cat = []
test_cat =[]
train_sub_cat = []
test_sub_cat = []

for rows in train_data['category']:
	train_cat.append(cat.index(rows))

for rows in train_data['sub_category']:
	train_sub_cat.append(sub_cat.index(rows))

for rows in test_data['sub_category']:
	test_sub_cat.append(sub_cat.index(rows))

for rows in test_data['category']:
	test_cat.append(cat.index(rows))

train_cat = tuple(train_cat)
train_sub_cat = tuple(train_sub_cat)
test_cat = tuple(test_cat)
test_sub_cat = tuple(test_sub_cat)

#classifier = LinearSVC()
classifier = MultinomialNB()
#classifier = BernoulliNB()
#classifier = SVC()
#classifier = KNeighborsClassifier(n_neighbors = 3, algorithm = 'auto')
#classifier = sklearn.neural_network.MLPClassifier()
#classifier = linear_model.SGDClassifier()

print "classificaion in category is going on"
classifier.fit(train_matrix,train_cat)

predict_cat = classifier.predict(test_matrix)
holdup_cat = classifier.predict(holdup_matrix)
#predicted_probs = classifier.predict_proba(test_matrix)

accuracy_cat = classifier.score(test_matrix,test_cat)
#precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(positive_test,predict_cat)

print "classification in category is done"
print ("accuracy = ",accuracy_cat)
#print ("precision =", precision)
#print ("recall =", recall)
#print ("f_score =",f1)
#print predict_cat
#print predicted_probs
cat_predict = []
cat_holdup =[]
for rows in predict_cat:
	cat_predict.append(cat[rows])
for rows in holdup_cat:
	cat_holdup.append(cat[rows])

print "Classification in sub category is going on"
classifier.fit(train_matrix,train_sub_cat)

predict_sub_cat = classifier.predict(test_matrix)
holdup_sub_cat = classifier.predict(holdup_matrix)
#predicted_probs = classifier.predict_proba(test_matrix)

accuracy_sub_cat = classifier.score(test_matrix,test_sub_cat)
#precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(positive_test,predict_cat)

print "classification in subcategory is done"
print ("accuracy = ",accuracy_sub_cat)
#print ("precision =", precision)
#print ("recall =", recall)
#print ("f_score =",f1)
#print predict_cat
#print predicted_probs
sub_cat_predict = []
sub_cat_holdup =[]
for rows in predict_sub_cat:
	sub_cat_predict.append(sub_cat[rows])
for rows in holdup_sub_cat:
	sub_cat_holdup.append(sub_cat[rows])



final_data = zip(test_data['product_name'],test_data['category'],cat_predict,test_data['sub_category'],sub_cat_predict)
with open(raw_input("output filename for predicted test_data>"),"wb") as csv_file:
	writer = csv.writer(csv_file, delimiter = ',')
	writer.writerow(("product_name","original_category","predicted_category","original_sub_category","predicted_sub_category"))
	for line in final_data:
		writer.writerow(line)

final_holdup_data = zip(holdup_data,cat_holdup,sub_cat_holdup)
with open(raw_input("output filename for predicted holdup_data>"),"wb") as csv_file:
	writer = csv.writer(csv_file, delimiter = ',')
	writer.writerow(("product_name","predicted_category","predicted_sub_category"))
	for line in final_holdup_data:
		writer.writerow(line)
cat_confusion_matrix= sklearn.metrics.confusion_matrix(test_data['category'],cat_predict,cat)

with open("cat_confusion_matrix1.csv","wb") as csv_file:
	writer = csv.writer(csv_file,delimiter = ',')
	writer.writerow(cat)
	for line in cat_confusion_matrix:
		writer.writerow(line)

sub_cat_confusion_matrix= sklearn.metrics.confusion_matrix(test_data['sub_category'],sub_cat_predict,sub_cat)

with open("sub_cat_confusion_matrix1.csv","wb") as csv_file:
	writer = csv.writer(csv_file,delimiter = ',')
	writer.writerow(sub_cat)
	for line in sub_cat_confusion_matrix:
		writer.writerow(line)