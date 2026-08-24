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
from multiprocessing import Pool
import multiprocessing as mp
from textcompressor import *

if not sys.warnoptions: #Suppresses warnings
    import warnings
    warnings.filterwarnings("ignore")

if __name__=='__main__':
    mp.freeze_support()

    #Uncomment for an interactive demonstration of various text quality and lengths, with a handicap of your choice
    #HANDICAP = 1.0 #Choose your handicap value!
    #textcompressor.demo(HANDICAP) 

    #Uncomment to sample your own text with a handicap of your own choice (WITHOUT parallel processing)
    #HANDICAP = 1.0 #Choose your handicap value!
    #text = open(<Insert your text file path here>, 'r', encoding='utf-8').read()
    #textcompressor.run_personal_demo(HANDICAP, text)

    #Runs the compressor (WITH parallel processing) on either the bundled Great Gatsby
    #sample or a .txt file of your own choosing.
    path = input('Enter a .txt file path to compress [default: The_Great_Gatsby.txt]... > ').strip()
    if not path:
        path = 'The_Great_Gatsby.txt'
    text = open(path, 'r', encoding='utf-8').read()
    summary = textcompressor.run(text)
    print(summary)

