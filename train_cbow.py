from gensim.models import Word2Vec
import gensim.downloader as api

# Download text8 dataset if not already present
dataset = api.load("text8")

# Train CBOW model on text8 with adjusted parameters
cbow_model = Word2Vec(
    sentences=dataset,
    vector_size=50,  # Reduced dimension for faster training and better generalization
    window=3,        # Smaller window for local context
    min_count=10,    # Ignore very rare words
    workers=4,       # Number of worker threads
    sg=0,            # CBOW architecture
    epochs=10,       # Increased epochs for better training
)
cbow_model.save("cbow_text8_optimized.model")
