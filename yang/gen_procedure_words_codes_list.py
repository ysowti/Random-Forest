
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

stopwords_set = set(stopwords.words('english'))
_tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

procedures = pd.read_csv('../procedures.csv')

words = set()
for p in procedures['procedure'].values:
    words.update(_tokenizer.tokenize(p.lower()))

words -= stopwords_set

feature_words = list(sorted(words))
proc_codes = sorted(procedures['procedure_code'].unique())



with open('procedure_words_list.csv', 'wt') as fid:
    fid.write('word\n')
    for w in feature_words:
        fid.write(w + '\n')

with open('procedure_codes_list.csv', 'wt') as fid:
    fid.write('code\n')
    for w in proc_codes:
        fid.write(w + '\n')