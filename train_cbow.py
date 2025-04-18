from gensim.models import Word2Vec
import numpy as np
from gensim.models.word2vec import Text8Corpus
import matplotlib.pyplot as plt

# Train CBOW on Wikipedia corpus (text8) ---

#Download text8 dataset if not already present
import gensim.downloader as api
dataset = api.load("text8")

# Train CBOW model on text8
cbow_model = Word2Vec(
    sentences=dataset,
    vector_size=100,  # Increased dimension for better representation
    window=5,         # Larger window for more context
    min_count=5,      # Ignore words that appear less than 5 times
    workers=4,        # More workers for faster training
    sg=0,            # CBOW architecture
    epochs=5         # Reduced epochs as we have much more data
)
cbow_model.save("cbow_text8.model")