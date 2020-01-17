#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1 Import Library
import pandas as pd
import numpy as np
import re
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[2]:


#2 Read the csv dataset
corpus = pd.read_csv('E:/Semester 7/NLP/Tugas Akhir/IMDB Dataset.csv')


# In[3]:


#3 Get the corpus length
print('Panjang Corpus : ',len(corpus))


# In[4]:


corpus.head()


# In[5]:


corpus.tail()


# In[6]:


#4 Count the positive and negative review
review_positif = len([x for x in corpus['sentiment'] if x == 'positive'])
review_negatif = len([x for x in corpus['sentiment'] if x == 'negative'])
print('Review Positif : ',review_positif)
print('Review Negatif : ',review_negatif)


# In[7]:


#5 data preprocessing
##1 html tag removal
bs_corpus = [BeautifulSoup(text).get_text() for text in corpus['review']]


# In[8]:


##2 punctual removal
import string
i=0
for word in bs_corpus:
    for punctuation in string.punctuation:
        word = word.replace(punctuation,"")
    for number in '1234567890':
        word = word.replace(number,"")
    bs_corpus[i] = word
    i = i+1

bs_corpus


# In[9]:


##3 case folding
cf_corpus = [x.lower() for x in bs_corpus]
cf_corpus


# In[10]:


##4 stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_corpus = [[ps.stem(word) for word in sentence.split(" ")] for sentence in cf_corpus]


# In[11]:


##5 lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_corpus = [[wordnet_lemmatizer.lemmatize(word) for word in sentence] for sentence in stemmed_corpus]


# In[12]:


lemmatized_corpus


# In[13]:


X = [" ".join(review) for review in lemmatized_corpus]
y = corpus['sentiment']


# In[14]:


X = np.asarray(X)


# In[15]:


##6 TF-Idf
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)
X_vect


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)


# In[17]:


##7 Naibe Bayes' Tokenization
class Tokenizer:
    
    #Remove HTML Tag
    def clean(self, text):
        no_html = BeautifulSoup(text).get_text()
        clean = re.sub("[^a-z\s]+", " ", no_html, flags=re.IGNORECASE)
        return re.sub("(\s+)", " ", clean)
    
    #Tokenizing Corpus
    def tokenize(self, text):
        clean = self.clean(text).lower()
        stopwords_en = stopwords.words("english")
        return [w for w in re.split("\W+", clean) if not w in stopwords_en]
        #stem tokenized word
        porter.stem(w)


# In[18]:


##8 Naive Bayes
class MultinomialNaiveBayes:
    def __init__(self, classes, tokenizer):
        self.tokenizer = tokenizer
        self.classes = classes
      
    def group_by_class(self, X, y):
        data = dict()
        for c in self.classes:
            data[c] = X[np.where(y == c)]
        
        return data
           
    def fit(self, X, y):
        self.n_class_items = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()

        n = len(X)
        
        grouped_data = self.group_by_class(X, y)
        
        for c, data in grouped_data.items():
            self.n_class_items[c] = len(data)
            self.log_class_priors[c] = math.log(self.n_class_items[c] / n)
            self.word_counts[c] = defaultdict(lambda: 0)
            
            for text in data:
                counts = Counter(self.tokenizer.tokenize(text))
                for word, count in counts.items():
                    if word not in self.vocab:
                        self.vocab.add(word)

                self.word_counts[c][word] += count
                
        return self
    
    def laplace_smoothing(self, word, text_class):
        num = self.word_counts[text_class][word] + 1
        denom = self.n_class_items[text_class] + len(self.vocab)
        
        return math.log(num / denom)
    
    def predict(self, X):
        result = []
        for text in X:
            class_scores = {c: self.log_class_priors[c] for c in self.classes}  
            words = set(self.tokenizer.tokenize(text))
            
            for word in words:
                if word not in self.vocab: continue
                for c in self.classes:
                    log_w_given_c = self.laplace_smoothing(word, c)
                    class_scores[c] += log_w_given_c
                
            result.append(max(class_scores, key=class_scores.get))

        return result


# In[19]:


##9 using MNB
MNB = MultinomialNaiveBayes(
    classes=np.unique(y), 
    tokenizer=Tokenizer()
).fit(X_train, y_train)


# In[20]:


y_predict_MNB = MNB.predict(X_test)


# In[21]:


cnf_matrix = confusion_matrix(y_test, y_predict_MNB)
cnf_matrix


# In[22]:


##10 using SVM
X_train_svm = vectorizer.transform(X_train)
X_test_svm = vectorizer.transform(X_test)

SVM = LinearSVC()
SVM.fit(X_train_svm, y_train)
y_predict_SVM = SVM.predict(X_test_svm)


# In[23]:


cnf_matrix = confusion_matrix(y_test, y_predict_SVM)
cnf_matrix


# In[24]:


##11 MNB
print("Accuracy: ", accuracy_score(y_test, y_predict_MNB))
print("Precision: ", precision_score(y_test, y_predict_MNB, average="weighted"))
print("Recall: ", recall_score(y_test, y_predict_MNB, average="weighted"))
print("F1_Score: ", f1_score(y_test, y_predict_MNB, average="weighted"))


# In[59]:


##13 Custom Test Input
print('Type your comment: ', end=" ")
test= []
test.append(input())
test_dtm = vectorizer.transform(test)


# In[60]:


predLabel = SVM.predict(test_dtm)


# In[61]:


print("Your review is", predLabel[0])


# In[144]:


test_Text = [X_test[7]]
test_Text


# In[145]:


test_Text_dtm = vectorizer.transform(test_Text)
predTextLabel = SVM.predict(test_Text_dtm)
print("Your review is", predTextLabel[0])

