from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
import os

current_dir = os.getcwd()
container = load_files(current_dir + '/features/')

# IMPLEMENT BAG OF WORDS

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf.fit(container.data, container.target)

test_feature_path = current_dir + '/mystery.txt'
test_doc = open(test_feature_path).read().decode('utf8').split('\n')
test_counts = CountVectorizer().transform(test_doc)
test_tfidf = TfidfTransformer().transform(test_counts)

predicted = text_clf.predict(test_tfidf)

for doc, category in zip(test_doc, predicted):
    print('%r => %s' % (doc, container.target_names[category]))

