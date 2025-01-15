"""Given Elmo embeddings, classifies text by it's genre."""
#Author: Greta Perez-Haiek (Jan 1st, 2024)
#For the purpose of optimizing StoryForge's user experience by providing auto-generated genres given a book.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA #For dimensionality reduction
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from elmo import * #for generating an embedding
from textcompressor import * #for generating an embedding
import pickle #for storing objects
import matplotlib.pyplot as plt
import time #timeing how long certain processes will take...


def get_data():
    '''Grabs data pickled from databuilder.py and converts it into "x" and "y" data arrays.'''
    #datasampler.pkl = 200 pts of randomly selected books... NOT balanced
    #data.pkl = full dataset... balanced, approx 1300 pts.
    df = pd.read_pickle('datasampler.pkl') #reads pickle file generated from "databuilder.py"

    df = df[df["elmo_embeddings"].apply(lambda x: not np.all(x == 0))] #removes datapoints where the embedding is all zeroes...

    #Grabbing the Genres... "y"!
    #Converts the dataframe "genres" into a list of lists of strings
    df["genres"] = df["genres"].apply(lambda x: x.strip("[]").replace("'", "").split(", "))
    mlb = MultiLabelBinarizer() #for multi-label prediction ease...
    mlb.fit((df["genres"].to_list()))
    y = mlb.transform((df["genres"].to_list()))

    #Grabbing the Embeddings... "X"!
    #pads the arrays so that it is all of equal size
    max_length = max(len(embedding) for embedding in df["elmo_embeddings"])
    df["elmo_embeddings"] = df["elmo_embeddings"].apply(lambda x: np.pad(x, (0, max_length - len(x)), 'constant'))
    X = np.vstack(df["elmo_embeddings"].values)

    with open('mlb.pkl','wb') as f:
        pickle.dump(mlb,f)
    return y, X, max_length

def visualize_genres(y):
    '''Given a y, plots the available genres (inclusive) of the given dataset!'''
    genres_list = ["20th Century", "Adventure", "Classics", "Fantasy", "Fiction", 
                "Historical", "Historical Fiction", "Literature", "Non-Fiction", "Romance"]

    #Assuming 'y' is a list of lists like [[0,0,1,1,0,0,0,0,0,0], ...]
    y_array = np.array(y)
    genre_counts = np.sum(y_array, axis=0) #counts each genre!

    #Lets see that beautiful data...
    plt.figure(figsize=(10, 6))
    plt.bar(genres_list, genre_counts, color='skyblue')
    plt.xlabel('Genre')
    plt.ylabel('Number of Datapoints')
    plt.title('Histogram of Genre Occurrences')
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()
    plt.show()
    return 0

def reduce_dimensions(X):

    #FOR PCA....
    # Assuming X is a numpy array of shape (~1300, ~3,000,000)
    # Step 1: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply PCA to reduce dimensions to 191 (limit = min(n_datapoints, n_features))
    pca = PCA(n_components=191)
    X_reduced = pca.fit_transform(X_scaled)

    with open("pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)

    # Check the shape of the reduced data
    print("Original shape:", X.shape)
    print("Reduced shape using PCA:", X_reduced.shape)

    #Check the varience
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Total PCA variance explained:", np.sum(explained_variance_ratio))

    return X_reduced

def get_embeddings_elmo(text):
    '''Given a text string, returns ELMO embeddings generated from Elmo.py.'''
    elmo = elmo_init()
    embeddings = []
    compressed = run(text) #compresses text for processing purposes...
    embedding = elmo_embed(elmo, compressed)
    embeddings.append(embedding['elmo'].numpy().flatten())
    return embeddings

def predict(text, model_name):

    #loads the model
    if model_name == "logistic":
        with open('logisticmodel.pkl', 'rb') as f:
            model = pickle.load(f)
            max_length = pickle.load(f)
    elif model_name == "svc":
        with open('svcmodel.pkl', 'rb') as f:
            model = pickle.load(f)
            max_length = pickle.load(f)

    #loads the multilabel binarizer
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)

    #loads PCA (to fit prediction into new dimensions)
    with open("pca_model.pkl", "rb") as f:
        pca = pickle.load(f)

    #generates a summary, then an embedding, given a text
    emb = get_embeddings_elmo(text)
    emb = emb[0]
    emb = np.pad(emb, (0, max_length - len(emb)), 'constant') 
    emb = emb.reshape(1, -1)
    emb = pca.transform(emb)

    #makes a prediction!
    pred = model.predict_proba(emb)
    pred = pred.tolist()
    pred = pred[0]
    print("BOOK: The Great Gatsby." )
    print("The Predictions for the given model is as follows..." ) #returns a prediction given a model and an embedding
    for i in range(len(pred)):
        print(f"{mlb.classes_[i]} :: {pred[i]*100:.4f} %")

def build_logistic():
    """Creates a logistic regression model with a One-Vs-The-Rest classifier approach (using Multi-Label Binarizer for
    Multi-Genre classification), then saves it for execution.
    
    Notes: Accuracy is lackluster. Logistical regression models do not function as well with feature lengths of 
    over 1,000 datapoints."""

    start_time = time.time()
    y, X, max_length = get_data()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time of Data Fetching: {elapsed_time:.6f} seconds")

    start_time = time.time()
    X = reduce_dimensions(X)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time of Dimension Reduction: {elapsed_time:.6f} seconds")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Creates a Logistic Regression model with OneVsRestClassifier....
    #The Higher the "C" value, the more closely the model will fit the data.

    model = OneVsRestClassifier(LogisticRegression(class_weight='balanced', C=0.4, max_iter = 10000)) #works but not too accurate

    #model.fit(X, y) #stalling right here... I should time it.

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  
    print(classification_report(y_test, y_pred, zero_division=0))

    with open('logisticmodel.pkl','wb') as f:
        pickle.dump(model, f)
        pickle.dump(max_length, f)

if __name__ == "__main__":
    #builds a logistic regression model and saves it as "logistic_regression_model.joblib"
    #also prints out the classification report while building
        
    if not sys.warnoptions: #Suppresses warnings3
        import warnings
        warnings.filterwarnings("ignore") 

    #Builds model... uncomment and run this ONCE
    #build_logistic()

    #Make an official prediction! Uncomment the sample text that you want to use
    #text = open('Carnegie_Loves_me_Draft.txt', 'r').read() #"Carnegie Loves me!" w/ a word count of ~6,000...'
    #text = open('Thundered_In_Act_One.txt', 'r').read() # "Thundered In: Act One" w/ a word count of ~9,000...'
    #text = open('Falon_Winters_Timeline.txt', 'r').read() # "Falon Winters: A Timeline" w/ a word count of ~10,000...'
    #text = open('AIR_Draft.txt', 'r').read() # "The AIR Models" w/ a word count of ~13,800...'
    text = open('The_Great_Gatsby.txt', 'r', encoding='utf-8').read() # "The Great Gatsby" w/ a word count of ~47,800...'
    #text = open('Beneath_the_Urban_Stars.txt', 'r', encoding='utf-8').read() # "Beneath the Urban Stars" by Beatrice...
    #text = open('GLOWROT.txt', 'r', encoding='utf-8').read() # "GLOWROT" by Beatrice...

    #Uncomment the model type that you want to use
    model_name = "logistic"
    #model_name = "svc"

    #Executes a prediction
    start_time = time.time()
    predict(text, model_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time of Prediction: {elapsed_time:.6f} seconds")
