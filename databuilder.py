"""This module builds the Project Gutenberg Dataset so that it can be utilized for training 
and classification purposes.

The following Dataset will be used:
(Sourced from Kaggle, Created from Project Gutenberg and GoodReads.com)
https://www.kaggle.com/datasets/michaelrussell4/10000-books-and-their-genres-standardized.

The dataset will be built from a .csv file (taken from the website above) and applied into a Pandas
dataframe, which will then be manipulated into a new dataframe of ELMO embeddings and associated genres.
There is an option in this module to visualize and CLEAN the dataset to verify that models can successfully train
itself upon it, although the following demo already demonstrated such. 
https://www.kaggle.com/code/michaelrussell4/gutenberg-book-genre-feature-engineering
I will be utilizing the pickled file in this open source repository for the analysis.

Dependencies:
Tensorflow - 
Numpy - 
Pandas - 
Langdetect - 
Pickle - 

Contains the following files: 
    {elmo.py}
"""
#Author: Greta Perez-Haiek (Last Updated: Nov 19th, 2024)
#For the purpose of optimizing StoryForge's user experience by providing auto-generated genres given a book.

import pandas as pd
import pickle
from elmo import *
from textcompressor import *
import numpy
import tensorflow
from tqdm import tqdm
import sys

if not sys.warnoptions: #Suppresses warnings3
    import warnings
    warnings.filterwarnings("ignore")

def load_dataframe():
    """To be executed once! Cleans up the publically available available dataset and saves it as a .cvs
    file for easy access. Removes all non-English, extremely long/short, and empty texts, along with extrenuous 
    genres. This way, the data that we will run the classification problem with is lean."""
    with open('clean_books_and_genres.p', 'rb') as f: #loads in cleaned pickled data as a pandas dataframe!
        cleandata = pickle.load(f)
        df = pd.DataFrame(cleandata)

    books_and_genres = pd.read_csv('books_and_genres.csv')

    #Extract columns from the CSV file
    books_and_genres = books_and_genres.iloc[:, [1, 2]]  #Assuming second column is titles, third is text
    books_and_genres.columns = ['title', 'text']  #Rename columns for clarity

    #Merge dataframes based on the 'title' column
    df_updated = pd.merge(df, books_and_genres, on='title', how='left')

    #Replace the NaN values in the 'text' column of the original dataframe
    df_updated['text'] = df_updated['text_y']
    df_updated = df_updated.drop(columns=['text_y'])  # Drop the additional column created during merge
    df_updated.rename(columns={'text_x': 'textnew'}, inplace=True)  # Rename back to original if needed
    df_updated= df_updated.iloc[:, list(range(4)) + list(range(-3, 0))]
    df_updated = df_updated.drop(columns=['textnew'], errors='ignore')

    pd.set_option('display.max_columns', None) 
    print(df_updated.head())
    print(df_updated.iloc[:, -1])

    #save the file as a .cvs
    df_updated.to_csv('books_and_genres_cleaned.csv', index=False)

def add_embeddings():
    df = pd.read_csv('books_and_genres_cleaned.csv', nrows=100) #read only the first 100 rows, for simplicity sake
    elmo = elmo_init()
    embeddings = []
    for _,s in enumerate(tqdm(df['text'])): 
        try:
            compressed = run(s) 
            embedding = elmo_embed(elmo, compressed)
            embeddings.append(embedding['elmo'].numpy().flatten())
        except:
            print("Embedding Generation Error Encountered!!")
            embeddings.append(numpy.zeros(283648,))
    try:    
        np_embeddings = np.array(embeddings)
        df['elmo_embeddings'] = np_embeddings
        df.to_csv('C:\\Users\\661994646\\Documents\\GitHub\\Book-To-Genre-using-ML\\books_genres_and_embeddings.csv', index=False)
    except:
        print("Saving to .CVS Error Encountered!!")

    breakpoint()

if __name__ == "__main__":
    #load_dataframe()
    add_embeddings()
