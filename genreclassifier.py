"""Given Elmo embeddings, classifies text by it's genre."""
#Author: Greta Perez-Haiek (Nov 19th, 2024)
#For the purpose of optimizing StoryForge's user experience by providing auto-generated genres given a book.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from joblib import dump #for saving a model
from joblib import load #for loading a model
from elmo import * #for generating an embedding
from textcompressor import * #for generating an embedding
from sklearn.impute import SimpleImputer

def logistic():
    "Creates a logistic regression model with a One-Vs-The-Rest classifier approach"

    #Creates data in the form of embeddings (x) and genre labels (y)
    #df = pd.read_csv('books_genres_and_embeddings.csv',skiprows=[1]) #reads in the embeddings dataframe created from databuilder.py
    #X = np.vstack(df["elmo_embeddings"].values)
    df = pd.read_pickle('data.pkl')  

    df["elmo_embeddings"] = pd.to_numeric(df['elmo_embeddings'], errors='coerce')
    X = np.vstack(df["elmo_embeddings"].values) #temporary demo solution, as ELMO embeddings is generating...
    y = MultiLabelBinarizer().fit_transform(df["genres"]) #this turns the genres into a binary basis matrix... makes it easier to learn!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Creates a Logistic Regression model with OneVsRestClassifier....
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    clf.fit(X_train, y_train) #train
    y_pred = clf.predict(X_test) #test
    print(classification_report(y_test, y_pred, target_names=mlb.classes_)) #evaluate

    #SAVE The trained model!
    dump(clf, "logistic_regression_model.joblib")
    return 0

def adaboost():
    """Creates an AdaBoost model with a One-Vs-The-Rest classifier approach and a Logistic Regression Base"""
    #Creates data in the form of embeddings (x) and genre labels (y)
    #df = pd.read_csv('books_genres_and_embeddings.csv') #reads in the embeddings dataframe created from databuilder.py
    #X = np.vstack(df["elmo_embeddings"].values)
    df = pd.read_pickle('data.pkl')  

    X = np.vstack(df["doc_vectr"].values) #temporary demo solution, as ELMO embeddings is generating...
    mlb = MultiLabelBinarizer()
    y = MultiLabelBinarizer().fit_transform(df["genres"]) #this turns the genres into a binary basis matrix... makes it easier to learn!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Create an AdaBoost model with OneVsRestClassifier...
    base_estimator = LogisticRegression(max_iter=1000, random_state=42)
    clf = OneVsRestClassifier(AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42))
    clf.fit(X_train, y_train) #train
    y_pred = clf.predict(X_test) #test
    print(classification_report(y_test, y_pred, target_names=mlb.classes_)) #evaluate

    #SAVE the Pre-Trained Model
    dump(clf, "adaboost_model.joblib")
    return 0

def predict(model, emb):
    "Given an embedding and a Model, creates a prediction and returns it."
    prediction = model.predict(emb) 
    return prediction

def get_embeddings_elmo(text):
    elmo = elmo_init()
    embeddings = []
    compressed = run(text) 
    embedding = elmo_embed(elmo, compressed)
    embeddings.append(embedding['elmo'].numpy().flatten())
    return embeddings

if __name__ == "__main__":
    #builds a logistic regression model and saves it as "logistic_regression_model.joblib"
    #also prints out the classification report while building
        
    if not sys.warnoptions: #Suppresses warnings3
        import warnings
        warnings.filterwarnings("ignore") 

    #get an example embedding
    text = open('Carnegie_Loves_me_Draft.txt', 'r').read() # "Carnegie Loves me!" w/ a word count of ~6,000...'
    emb = get_embeddings_elmo(text)

    logistic() 
    loaded_model = load("logistic_regression_model.joblib")
    print("The logistic regression pred is..." ,predict(loaded_model, emb)) #returns a prediction given a model and an embedding

    #builds an Adaboost regression model and saves it as "adaboost_model.joblib"
    #also prints out the classification report while building
    #adaboost()
    #loaded_model = load("adaboost_model.joblib")
    #print("The Adaboost regression pred is..." ,predict(loaded_model, emb)) #returns a prediction given a model and an embedding

    #Compare predictions... is there a difference? Choose the one with the better predictions and classification report!


