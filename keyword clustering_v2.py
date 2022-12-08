import pandas as pd
import numpy as np
import openpyxl as op
import csv

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster

pd.set_option("display.max_rows", None, "display.max_columns", None)

stemmer = PorterStemmer()
sw = stopwords.words('english')

def tokenizer(keyword):
    return [stemmer.stem(w) for w in keyword.split(' ')]

input_file = input("Please, enter the excel file name (without extension) : ")

print("Output file name will be = Output.csv")

wb = op.load_workbook(input_file+".xlsx")
sh = wb.active
keywords = []


for row in range(2, 1001):
    keywords.append(sh['b'+str(row)].value)
    # Keywords are being taken from column B of excel file
wb.close()



tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=sw)
X = pd.DataFrame(tfidf.fit_transform(keywords).toarray(),
                 index=keywords, columns=tfidf.get_feature_names_out())
c = cluster.AffinityPropagation()
X['pred'] = c.fit_predict(X)
header = ['pred']
X.to_csv('Output.csv', columns=header)
#print(X['pred'])
