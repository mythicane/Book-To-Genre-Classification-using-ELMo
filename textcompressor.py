"""When given a long text input, compresses the text into a significantly 
shorter version by it's most "important sentences" using the Natural Language
Processing Tooklit and Scikit-Learn's TfidfVectorizer algorithm, which measures 
the importance of each individual word in a text.

The Natural Language Processing Toolkit allows access to a small portion of the Gutenberg dataset, 
as well as the Brown Dataset (which is a set of 500 open-sourced texts of a combined +1 million words,
all taged by the following genres): 
    
    ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies',
'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance',
'science_fiction']
    
While primitive, The Brown Dataset was created for "stylistics" research initiatives, which involves
analyzing the correlation of certain words with it's associated genre. (See https://www.nltk.org/book/ch02.html 
for more information on the various NLTK corpuses that we can use!) 

Scikit-Learn's TfidfVectorizer algorithm is considered a primitive form of NLP, and is NOT generative
in nature. It is broken down into two analytical processes: TF (Term Frequency), which estimates how frequenty a 
word occurs in a text... and IDF (Inverse Document Frequency), which estimates how important a lexeme (or string 
of thoughts) is to the context of a text. Both results are multiplied together to give each lexeme a "score," and 
entire sentences (potentially containing one or multiple lexemes) will be scored against each other to choose 
which portions of the text to keep and which portions to throw away.

Know this: This is NOT a text-summarizer, as it does not generate new text, but compress from pre-existing text.

This file is inspired by the following approach:
https://towardsdatascience.com/text-summarization-using-tf-idf-e64a0644ace3
https://medium.com/saturdays-ai/building-a-text-summarizer-in-python-using-nltk-and-scikit-learn-class-tfidfvectorizer-2207c4235548

Differences between this approach and this programming include using Scikit-Learn for the TfidVector algorithm
and utilizing NLTK for access to the dataset for demonstration purposes.

Dependencies:
    
NLTK (https://www.nltk.org/data.html)
Gutenberg-PY (https://github.com/raduangelescu/gutenbergpy)
Gutenberg (https://pypi.org/project/Gutenberg/)
Berkeley DB (Licensing for Gutenberg Dataset)
"""
#Author: Greta Perez-Haiek (Last Updated: Nov 19th, 2024)
#For the purpose of optimizing StoryForge's user experience by providing auto-generated genres given a book.

import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
from nltk.corpus import stopwords, gutenberg, wordnet, brown
import multiprocessing #for optimization and speed efforts
import sys #for the suppression of warnings
import time
from multiprocessing import Pool
import threading

summaries = [] #the global variable!

if not sys.warnoptions: #Suppresses warnings3
    import warnings
    warnings.filterwarnings("ignore")

def remove_punctuation_marks(text):
    """When given a text (as a string), returns the same text, but without punctuation marks."""
    punctuation_marks = dict((ord(punctuation_mark), None) for punctuation_mark in string.punctuation)
    return text.translate(punctuation_marks)

def get_lemmatized_tokens(text):
    """Given a text string, splits the text into 'lexemes', which is a string of thoughts. The algorithm
    will treat these lexemes as it's own "word," as sometimes the same words in different sentences will have
    different meanings and importance in a book."""
    normalized_tokens = nltk.word_tokenize(remove_punctuation_marks(text.lower()))
    return [nltk.stem.WordNetLemmatizer().lemmatize(normalized_token) for normalized_token in normalized_tokens]

def get_average(values):
    """Returns the average "value" of a lexeme in a text string."""
    greater_than_one_count = total = 1
    for value in values :
        if value != 0 :
            greater_than_one_count += 1
            total += value 
    return total / greater_than_one_count

def get_threshold(tfidf_results):
    """ Generates the average value of a 'sentence'- later in the algorithm, if a sentence falls below this
    threshhold, it will be eliminated in the summary as it's importance is not high enough, contextually."""
    i = total = 0
    while i < (tfidf_results.shape[0]):
        total += get_average(tfidf_results[i, :].toarray()[0])
        i += 1
    return total / tfidf_results.shape[0]

def get_summary(documents, tfidf_results, HANDICAP):
    """Given the handicap, documents (a.k.a, sentences in a text), and the vectorization results 
    and "value" thresholds of a lexeme", returns the summary string as a function of the three."""
    summary = ""
    i = 0
    while i < (tfidf_results.shape[0]):
        if (get_average(tfidf_results[i, :].toarray()[0])) >= get_threshold(tfidf_results) * HANDICAP : summary += ' ' + documents[i]
        i += 1
    return summary

def demo(HANDICAP):
    """Runs a simple demonstration of the algorithm with a given Handicap against a sample of a
    novel derived from the NLTK Open Source Brown Dataset, OR a less-professional "hobby" text by
    Greta Perez-Haiek."""

    print("******************************************")
    print("Text-Compressor DEMO running...\n")
    print("******************************************")
    print("Hello, welcome to the Storyforge Text Compressor Demonstration!")
    response = input("Would you like to review fan-made literature or classical literature? [f/c]...> ")
    if response == "f": #for less professional texts
        while True:
            print("Specify the book that you would like to compress...")
            print("")
            print('[1] - Genre: Tragiromance...	Greta\'s "Carnegie Loves me!" (6,000 words)')
            print('[2] - Genre: Action Science Fiction Screenplay...	Greta\'s "Thundered In - ACT ONE" (9,000 words)')
            print('[3] - Genre: Action-Packed Space Opera...	Greta\'s "Falon Winters" (10,000 words)')
            print('[4] - Genre: Slice of Life Sci-Fi...	Greta\'s "The A.I.R Models" (13,000 words)')
            print('[5] - Genre: Classical Literature...	"The Great Gatsby" (48,000 words)')
            num = input('Choose a number from the above... > ')
            if num == str(1):
                text = open('Carnegie_Loves_me_Draft.txt', 'r').read() # "Carnegie Loves me!" w/ a word count of ~6,000...'
            elif num == str(2):
                text = open('Thundered_In_Act_One.txt', 'r').read() # "Thundered In: Act One" w/ a word count of ~9,000...'
            elif num == str(3):
                text = open('Falon_Winters_Timeline.txt', 'r').read() # "Falon Winters: A Timeline" w/ a word count of ~10,000...'
            elif num == str(4):
                text = open('AIR_Draft.txt', 'r').read() # "The AIR Models" w/ a word count of ~13,800...'
            elif num == str(5):
                text = open('The_Great_Gatsby.txt', 'r', encoding='utf-8').read() # "The Great Gatsby" w/ a word count of ~47,800...'
            else:
                print("Invalid Number! Restarting Demo...")
                continue
            print("")
            print('TESTING WITH THE FAN-TEXT SELECTED...')
            print("")
            print("******************************************")
            print("The origional text is the following....")
            print('')
            print(text)
            print("")
            print("The End.")
            print("******************************************")
            print("The condensed text is the following....")
            print('')
            print(run_personal_demo(HANDICAP, text))
            print('')
            print("The End.")
            print("******************************************")
            print("DEMONSTRATION CONCLUDED. Would you like to try another Fan Written text?")
            decision = input('[y/n]...> ')
            if decision == "y":
                continue
            else:
                print("Terminating demonstration.")
                break
    elif response == "c": #for more professional literature
        while True:
            print("")
            print("Specify the book that you would like to compress...")
            print("")
            print('[1] - Genre: Mystery...	Hitchens\'s "Footsteps in the Night"')
            print('[2] - Genre: Science Fiction...	Heinlein\'s "Stranger in a Strange Land"')
            print('[3] - Genre: Adventure...	Field\'s "Rattlesnake Ridge"')
            print('[4] - Genre: Romance...	Callaghan\'s "A Passion in Rome"')
            print('[5] - Genre: Humor...	Thurber\'s "The Future, If Any, of Comedy"')
            num = input('Choose a number from the above... > ')
            if num == str(1):
                text = ' '.join(brown.words(fileids=['cl13']))
            elif num == str(2):
                text = ' '.join(brown.words(fileids=['cm01']))
            elif num == str(3):
                text = ' '.join(brown.words(fileids=['cn15']))
            elif num == str(4):
                text = ' '.join(brown.words(fileids=['cp12']))
            elif num == str(5):
                text = ' '.join(brown.words(fileids=['cr06']))
            else:
                print("Invalid Number! Restarting Demo...")
                continue
            print("")
            print('TESTING WITH THE BROWN DATASET NOVEL SELECTED...')
            print("")
            print("******************************************")
            print("The origional text is the following....")
            print('')
            print(text)
            print("")
            print("The End.")
            print("******************************************")
            print("The condensed text is the following....")
            print('')
            print(run_personal_demo(HANDICAP, text))
            print('')
            print("The End.")
            print("******************************************")
            print("DEMONSTRATION CONCLUDED. Would you like to try another novel?")
            decision = input('[y/n]...> ')
            if decision == "y":
                continue
            else:
                print("Terminating demonstration.")
                break
    else:
        print("Invalid Response! Terminating demonstration.")
    return 0 #returns null

def process_chunk(chunk):
        """Processes a single chunk of text to generate its summary."""
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
        documents = nltk.sent_tokenize(chunk)
        HANDICAP = 1.2 #THIS HANDICAP WAS CHOSEN BECAUSE IT WORKS TE BEST WITH TEXTS APPROX. 2,000 WORDS IN LENGTH!! (See documentation)
        tfidf_results = TfidfVectorizer(tokenizer=get_lemmatized_tokens, stop_words=stopwords.words('english')).fit_transform(documents)
        return get_summary(documents, tfidf_results, HANDICAP)

def run_personal_demo(HANDICAP, text):
    """When given text, returns the summarized text as a String.
    Ths function is for texts that are shorter in nature, with a
    user choosen handicap value. Best for demonstrations, experiments,
    and verification/validation procceses."""
    documents = nltk.sent_tokenize(text)
    tfidf_results = TfidfVectorizer(tokenizer=get_lemmatized_tokens, stop_words=stopwords.words('english')).fit_transform(documents)
    return get_summary(documents, tfidf_results, HANDICAP)

def run(text):
    """When given text, returns the summarized text as a String.
    Ths function is for texts that are longer in nature, and hence, needs splicing
    with the "chunking" option. This function does not intake a handicap value,
    as with a pretetermined chunk os 2,000 (with a processing speed of 4,000 words/sec),
    aa handicap of 1.2 is best. Best for testing, trials, and parallel procceses."""
    #splice the text into equal word chunks, ensuring each chunk has a maximum of 2000 words.
    words = text.split() #split a text by its words...
    if len(words) > 2000:
        chunks = [' '.join(words[i:i + 2000]) for i in range(0, len(words), 2000)]
    else:
        chunks = [text]  #If the text is less than 2000 words, process it as a single chunk
        
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        summaries = pool.map(process_chunk, chunks)
 
    #Combine the summaries of all the chunks to get a mega-summary.
    mega_summary = ' '.join(summaries)

    #if the mega-summary exceeds 5,000 words, rerun the program until the summary is 10,000 words...
    while len(mega_summary.split()) > 4000:
        mega_summary = run(mega_summary)  #recursively reprocess the mega-summary
    return mega_summary #returns the summary as a string!