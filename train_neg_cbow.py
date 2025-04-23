from gensim.models import Word2Vec
import numpy as np
from gensim.models.word2vec import Text8Corpus
import matplotlib.pyplot as plt

# Train CBOW on Wikipedia corpus (text8) ---
import gensim.downloader as api
dataset = api.load("text8")

# Train CBOW model on text8 with negative sampling
cbow_model = Word2Vec(
    sentences=dataset,
     vector_size=50,  # Reduced dimension for faster training and better generalization
     window=5,        # Smaller window for local context
     min_count=10,    # Ignore very rare words
     workers=4,       # Number of worker threads
     sg=0,            # CBOW architecture
     epochs=10,       # Increased epochs for better training
     negative=20,      # Commonly used value for negative sampling
)
cbow_model.save("cbow_negsample_text8.model")
