import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import nltk

ps=PorterStemmer()
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def transform_text(msg):
    msg=msg.lower()  #convert all letters to lower case
    msg=nltk.word_tokenize(msg) #break to tokens
    y=[] #to remove special char
    for i in msg:
        if i.isalnum():
            y.append(i)
    
    msg=y[:]
    y.clear() #to remove stop words and punctuation
    for i in msg:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    msg=y[:]
    y.clear()

    for i in msg: #stemming
        y.append(ps.stem(i))
    return " ".join(y)
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("email/sms spam classifier")
input_sms=st.text_input("enter the msg")
if st.button('Predict'):
# 1. preprocess
  transform_sms=transform_text(input_sms)
# 2. vectorize
  vector_input=tfidf.transform([transform_sms])
# 3. predict
  result=model.predict(vector_input)[0]
# 4. display
  if result==1:
    st.header("Spam :(")
  else:
    st.header("Not Spam :)")