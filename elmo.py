"""Builds and Freezes an Elmo Object for NLP purposes. When given a text, an ELMo network predicts 
the genre associated with the text. 

ELMo is intended to be be trained using the Project Gutenberg Dataset (see databuilder.py) and it's associated genres, from GoodReads.com.
The model will then be frozen so that backpropagation is halted. This will make the model lighter to use and handle.

Previously, I conducted research on BERT, a Transformer prececessor to GTP-2, and plenty of papers 
expressed BERT'S potential in language contextualization.This project wanted to use BERT Model
(https://en.wikipedia.org/wiki/BERT_%28language_model%29), as it's an advanced NLP model... However,
research shed light onto the fact that the open-source BERT model was scraped from indie authors 
without their consent (https://en.wikipedia.org/wiki/BookCorpus). Because this violate's StoryForge's 
commitment on ethically sourcing their data, I cannot in good conscience use BERT for this task.

We need to use a more primitive model than BERT, as the GTP models that came after BERT were also trained on
web-scrapped data.

Before BERT, there was ELMo (https://arxiv.org/pdf/1802.05365), a Bi-Directional Long Short Term Memory 
Algorithm trained on News Stories translated from multiple languages by a volunteer panel of scientists
looking to generate ethical language data for NLP training (https://arxiv.org/pdf/1312.3005) back in 2011. 
Because ELMo is the most advanced NLP algorithm that fits StoryForge's data-sourcing Standard, I will be using
a simpler version of ELMo for the task at hand!

Further Details on ELMo's Source Training data:
https://www.statmt.org/lm-benchmark/
https://statmt.org/wmt11/translation-task.html

ELMo will be accessed using Tensorflow Hub: https://www.tensorflow.org/hub.

Dependencies:
Tensorflow - https://www.tensorflow.org/api_docs/python/tf/compat/v1
Numpy - https://numpy.org/
Pandas - 

Contains the following files: 
    {databuilder.py}

"""
import tensorflow.compat.v1 as tf
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import databuilder
import sys

if not sys.warnoptions: #Suppresses warnings3
    import warnings
    warnings.filterwarnings("ignore")

def elmo_init():
    '''Returns an Elmo Object'''
    elmo = hub.load("https://tfhub.dev/google/elmo/3") #trainable true = live, trainable false = FROZEN!   
    return elmo

def elmo_embed(elmo, s):
    ''' Given an Elmo Object and a String "S", returns Elmo-Generated string embeddings.'''
    string_tensor = tf.constant([s])
    return elmo.signatures["default"](string_tensor)
    
if __name__ == "__main__":
    elmo = elmo_init()
    embedding = elmo_embed(elmo, "Hello World!")
    print(embedding)