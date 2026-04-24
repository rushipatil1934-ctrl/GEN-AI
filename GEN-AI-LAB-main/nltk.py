import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import counter

#download required NLTK data(run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordlet')
nltk.download('averaged_perceptron_tagger')

