import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template,request 


app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def posttext():
    text = request.form['text']
    processed_text = text.upper()
    #str='the best food that i had'
    X_test[0]=text
    tes=count_vect.transform(X_test)
    pred1 = model_random.predict(tes)
    global result    
    result=pred1[0]
    return render_template('positive.html',result=pred1[0])
      
@app.before_first_request
def loadthemode():
 
 
  data = pd.read_csv('Reviews.csv')
  data = data[data.HelpfulnessNumerator <= data.HelpfulnessDenominator]
  data['Score'] = data["Score"].apply(lambda x: "positive" if x > 3 else "negative")
  sorted_data = data.sort_values('ProductId',axis = 0, inplace = False, kind = 'quicksort',ascending = True)
  filtered_data = sorted_data.drop_duplicates(subset = {'UserId','ProfileName','Time'} ,keep = 'first', inplace = False)
  filtered_data['Score'].value_counts()
  final = filtered_data.copy()
  nltk.download('stopwords')
  stop = set(stopwords.words("english"))
  st = PorterStemmer()
  st.stem('burned')
  def cleanhtml(sent):
      cleanr = re.compile('<.*?>')
      cleaned = re.sub(cleanr,' ',sent)
      return cleaned
  def cleanpunc(sent):
      clean = re.sub(r'[?|!|$|#|\'|"|:]',r'',sent)
      clean = re.sub(r'[,|(|)|.|\|/]',r' ',clean)
      return clean
  i=0
  all_positive_reviews =[]
  all_negative_reviews = []
  final_string = []
  stem_data = " "
  for p in final['Text'].values:
      filtered_sens = []#filtered word
      p = cleanhtml(p)
      for w in p.split():
       # print(w)
          punc = cleanpunc(w)
          for s in punc.split():
            #print(w)
              if (s.isalpha()) & (len(s)>2):
                  if s.lower() not in stop:
                      stem_data = (st.stem(s.lower())).encode('utf8')
                    #can we use lemmatizer and stemming altogether??
                      filtered_sens.append(stem_data)
                      if (final['Score'].values)[i] == 'positive':
                          all_positive_reviews.append(stem_data)
                      if (final['Score'].values)[i] == 'negative':
                          all_negative_reviews.append(stem_data)
                  else:
                      continue
              else:
                  continue
    #print(filtered_sens)
      str1 = b" ".join(filtered_sens)
    #print(str1)
      final_string.append(str1)
      i+=1
  final['CleanedText'] = final_string
  final = final.sort_values('Time',axis= 0,inplace = False , na_position = 'last',ascending = True) 
  global X
  X = final['CleanedText'].values
  X = X[:100000]
  global y
  y = final['Score'].values
  y = y[:100000]
  global X_train ,X_test,y_train,y_test
  X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y)
  global count_vect
  count_vect = CountVectorizer() #in scikit-learn
  bow_train = count_vect.fit_transform(X_train)
  bow_test = count_vect.transform(X_test)
  x = np.random.normal(loc = 0 , scale = 0.1,size = 50)
  param_distb =  {'C': [y for y in x if y >0  ]}
  global model_random
  model_random = RandomizedSearchCV(LogisticRegression(class_weight = 'balanced',penalty = 'l1'),param_distb,cv = 10 ,scoring = 'accuracy')
  model_random.fit(bow_train,y_train)
  pred = model_random.predict(bow_test)
  model_random = RandomizedSearchCV(LogisticRegression(class_weight = 'balanced',penalty = 'l1'),param_distb,cv = 10 ,scoring = 'accuracy')
  model_random.fit(bow_train,y_train)
  pred = model_random.predict(bow_test)
  """str='the best food that i had'
  X_test[0]=str
  tes=count_vect.transform(X_test)
  pred1 = model_random.predict(tes)
  pred1[0]"""
	
	
	
if __name__ == '__main__':
    app.run(debug=True)
	
	
	
	
	




