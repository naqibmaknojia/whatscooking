import json
import io
import unicodedata
import sys
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

DIAGNOSTICS = True

def strip_accents(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn')
		
		
class LemmaTokenizer(object):
	def __intit__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self , doc):
		return [ self.wnl.lemmatize(t) for t in word_tokenize(doc)]
		
#load data (train_small.json, train.json)
train_file = io.open('train.json', 'r')
train_json = json.loads(strip_accents(train_file.read()))
test_file = io.open('test.json', 'r')
test_json = json.loads(strip_accents(test_file.read()))

i = 0

X1 = []
Y1 = []
X1s = []
Y1s = []
for o in train_json:
	X1.append(" ".join(o['ingredients']))
	Y1.append(o['cuisine'])
	
id_test = []
X2 = []
for o in test_json:
	id_test.append(o['id'])
	X2.append(" ".join(o['ingredients']))
	
# Print diagnostics
if DIAGNOSTICS:
	print '%d = %d' % (len(X1), len(Y1))
	print '%s : %s' % (X1[0], Y1[0])
	print '%d = %d' % (len(X2), len(id_test))

# TfidfVectorizer
print "vectonizer"
sys.stdout.flush()
vect = TfidfVectorizer(tokenizer=LemmaTokenizer()).fit(X1)
tf1 = vect.transform(X1).todense()
tf1s = vect.transform(X1s).todense()

tf2 = vect.transform(X2).todense()

#Train
print "train"
sys.stdout.flush()

c1f1 = RandomForestClassifier(max_depth=40, n_estimators=20).fit(tf1,Y1)
c1f3 = SGDClassifier(loss='modified_huber', penalty='12' , alpha=le-4 , n_iter=5, random_state=65).fit(tf1,Y1)
c1f = VotingClassifier(estimators=[('dt', clf1), ('sgd', clf3)], voting='soft' , weights=[1, 3]).fit(tf1 , Y1)

#Predict 
print "predict"
sys.stdout.flush()
Y1_hat = clf.predict(tf1)
Y2_hat = clf.predict(tf2)

#print daignostics
if DIAGNOSTICS:
	#print vect.vocabulary
	print "Empirical accuracy: %f" % np.mean(Y1_hat == Y1)
	#print metrics.classification_report(Y1, Y1_hat)
	
#Output predictions
out = io.open( 'submission.csv', 'w')
out.write(u'id,cuisine\n')
for i in range(len(X2)):
	out.write('%s,%s\n' % (id_test[i], Y2_hat[i]))










