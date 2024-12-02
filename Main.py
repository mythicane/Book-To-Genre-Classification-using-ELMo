"""Author: Greta Perez-Haiek (Last Updated: Nov 19th, 2024)
#For the purpose of optimizing StoryForge's user experience by providing auto-generated genres given a book.

Contains the following files: 
    {main.py
    textcompressor.py
    summarytogenre.py
    databuilder.py
    elmo.py}

Hyperparameters:
    
    {Handicap: A Constant
     Text: A string}
    
Machine Learning Models, Datasets, and Resources used:
    {Natural Language Tool-Kit (NLTK) - Word Tokenizer, Brown Dataset, Project Gutenberg Dataset, StopWords Dataset, Wordnet Dataset
     Scikit-Learn - TfidfVectorizer Algorithm
     ALLEN NLP - ELMo ("Embeddings from Language Model", a type of Bi-directional Long Short Term Memory Network)
     Tensorflow - Keras, Hub
     Kaggle, Project Gutenberg, GoodReads.com - "10,000 Books and Their Genres *standardized*" Dataset
     }
    
    """
import textcompressor
import elmo as elmo_module
from nltk.corpus import brown
import sys #for the suppression of warnings
import numpy as np
import pandas as pd

if not sys.warnoptions: #Suppresses warnings
    import warnings
    warnings.simplefilter("ignore")

###########################################
#TEXT COMPRESSOR FUNCTION CALLS
HANDICAP = 1.0 #Choose your handicap value!

#Uncomment for an interactive demonstration of various text quality and lengths, with a handicap of your choice
#textcompressor.demo(HANDICAP) 

#Uncomment to sample your own text with a handicap of your own choice (without parallel processing)
#textcompressor.run_personal_demo(HANDICAP, text) 

#Uncomment to sample your own text (with parallel processing)
#textcompressor.run(text)
###########################################

###########################################
#CLASSIFIER FUNCTION CALLS (Featuring elmo!)

if __name__ == '__main__':
    corpus = [
             ' '.join(brown.words(fileids=['cl13'])),
             ' '.join(brown.words(fileids=['cm01'])),
             ' '.join(brown.words(fileids=['cn15'])),
             ' '.join(brown.words(fileids=['cp12'])),
             ' '.join(brown.words(fileids=['cr06']))
    ]

    elmo = elmo_module.init()
    for i,raw_text in enumerate(corpus):
        compressed_text = textcompressor.run(HANDICAP, raw_text) #this will formally run the compressor
        embedding = elmo_module.embed(elmo, compressed_text)
        print("******************************************")
        print("\nRaw text len: \n",len(raw_text))
        print("******************************************")
        print("\nCompressed text: \n",len(compressed_text))
        print("******************************************")
        print("\nembeddeding: \n",embedding)
        print("******************************************")
###########################################