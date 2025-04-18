from gensim.models import Word2Vec
import numpy as np
from gensim.models.word2vec import Text8Corpus
import csv
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt


# --- Define garden path and control sentences ---

#parse lines in garden path dataset then convert it into format as test_data
test_data = []

# Read and process the gardenpath_dataset.csv file
csv_file_path = "/Users/zhudian/Desktop/21366 research/gardenpath_dataset.csv"
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header line
    for row in csv_reader:
        
        sentence = [word.rstrip('.,') for word in row[0].lower().split()]   # Convert to lowercase and split the sentence into words
        target_index = int(row[1])  # Convert target index to integer
        label = row[2]  # Get the label
        test_data.append((sentence, target_index, label))

# ---  Enhanced evaluation function ---
def evaluate_prediction_with_rank(model, sentence, target_index):
    # Convert all words to lowercase for consistency
    sentence = [word for word in sentence]
    target_word = sentence[target_index]
    
    # Collect context words within window size 2
    context_words = []
    window_size = 2
    for i in range(max(0, target_index - window_size), target_index):
        context_words.append(sentence[i])
    for i in range(target_index + 1, min(len(sentence), target_index + window_size + 1)):
        context_words.append(sentence[i])
    
    # Check if target word and context words are in vocabulary
    if target_word not in model.wv:
        print(f"Warning: Target word '{target_word}' not in vocabulary")
        return None
    
    context_vectors = []
    for word in context_words:
        if word in model.wv:
            context_vectors.append(model.wv[word])
        else:
            print(f"Warning: Context word '{word}' not in vocabulary")
    
    if not context_vectors:
        print(f"Warning: No valid context words for sentence at index {target_index}")
        return None

    # Calculate context vector and similarities based on word type

    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

    # Map NLTK POS tags to WordNet POS tags
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    # Tag words in the vocabulary with their POS
    vocab_pos = {word: get_wordnet_pos(tag) for word, tag in pos_tag(model.wv.index_to_key)}

    # Get the POS of the target word
    target_pos = get_wordnet_pos(pos_tag([target_word])[0][1])

    # Calculate context vector and similarities for words of the same type
    context_vector = np.mean(context_vectors, axis=0)
    similarities = []
    for word in model.wv.index_to_key:
        word_pos = vocab_pos.get(word)
        if word_pos == target_pos:  # Only consider words of the same type
            sim = np.dot(context_vector, model.wv[word]) / (
                np.linalg.norm(context_vector) * np.linalg.norm(model.wv[word]))
            similarities.append((word, sim))
    
    # Sort by similarity and get rank
    ranked_words = sorted(similarities, key=lambda x: -x[1])
    target_rank = next((i + 1 for i, (word, _) in enumerate(ranked_words) 
                       if word == target_word), None)
    
    return target_rank

# --- Run evaluation with detailed statistics ---
model = Word2Vec.load("cbow_text8.model")
results = {"garden": [], "control": []}
failed_cases = {"garden": [], "control": []}

for sentence, idx, label in test_data:
    rank = evaluate_prediction_with_rank(model, sentence, idx)
    if rank is not None:
        results[label].append(rank)
    else:
        failed_cases[label].append(" ".join(sentence))
    print(f"Sentence: {' '.join(sentence)} | Target Index: {idx} | Label: {label} | Rank: {rank if rank is not None else 'Failed'}")

# Calculate and display statistics
for label in ["garden", "control"]:
    avg_rank = np.mean(results[label]) if results[label] else float('inf')
    median_rank = np.median(results[label]) if results[label] else float('inf')
    print(f"\n{label.capitalize()} Path Results:")
    print(f"Average Rank: {avg_rank:.2f}")
    print(f"Median Rank: {median_rank:.2f}")
    print(f"Sample Size: {len(results[label])}")
    print(f"Failed Cases: {len(failed_cases[label])}")

# --- Plot results ---
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['Garden Path', 'Control']
avg_ranks = [np.mean(results["garden"]) if results["garden"] else 0,
            np.mean(results["control"]) if results["control"] else 0]

bars = ax.bar(labels, avg_ranks, color=['red', 'green'])
ax.set_ylabel('Average Rank (Lower is Better)')
ax.set_title('CBOW Model Performance: Garden Path vs Control Sentences')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom')

plt.show()
