import re
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer  ,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix , accuracy_score

data = pd.read_csv("SMSSpamCollection.txt" , sep = "\t" , names = ["label" , "message"])
print(data)

wordnet = WordNetLemmatizer()
corpus = []
for i in range(0 , len(data)):
    review = re.sub('[^a-zA-Z]' , ' ' , data["message"][i])
    review  = review.lower()
    review = review.split()

    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

print(corpus)

cv = CountVectorizer(max_features=2500)
independent_x = cv.fit_transform(corpus).toarray()

dependent_y = pd.get_dummies(data["label"] , drop_first=True)
print(independent_x)
print(dependent_y)

train_x , test_x , train_y , test_y = train_test_split(independent_x , dependent_y, test_size=0.20 , random_state=0)

span_model = MultinomialNB()
span_model.fit(train_x , train_y)

y_pred = span_model.predict(test_x)


print(confusion_matrix(test_y , y_pred))
print(accuracy_score(test_y , y_pred))



