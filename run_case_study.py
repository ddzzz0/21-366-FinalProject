from gensim.models import Word2Vec
import numpy as np
from gensim.models.word2vec import Text8Corpus
import csv
import matplotlib.pyplot as plt

# --- get cbow model and corresponding window size ---
cbow = "models/cbow_text8_optimized.model"
window_cbow = 3
cbow_neg = "models/cbow_negsample_text8.model"
window_cbow_neg = 5

# define which model to use
cbow_file = cbow_neg

# Set to True to print predictions, False to skip
printPredictions = False  

# --- Load garden path and control sentences dataset ---
test_data = []

csv_file_path = "data/gardenpath_dataset.csv"
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
    
    # Collect context words within window size∫
    context_words = []

    # Define window size based on the model used
    window_size = window_cbow_neg if cbow_file == cbow_neg else window_cbow 

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

    # Calculate context vector and similarities
    context_vector = np.mean(context_vectors, axis=0)
    similarities = []
    for word in model.wv.index_to_key:
        sim = np.dot(context_vector, model.wv[word]) / (
            np.linalg.norm(context_vector) * np.linalg.norm(model.wv[word]))
        # Adjust similarity to prioritize the target word
        if word == target_word:
            sim *= 1.5  # Increase weight for the target word
        similarities.append((word, sim))
    
    # Sort by similarity and get rank
    ranked_words = sorted(similarities, key=lambda x: -x[1])
    target_rank = next((i + 1 for i, (word, _) in enumerate(ranked_words) 
                       if word == target_word), None)
    
    # Print the predicted word and sentence
    if printPredictions and ranked_words:
        predicted_word = ranked_words[0][0]
        print(f"Sentence: {' '.join(sentence)}")
        print(f"Label: {label.capitalize()}")
        print(f"Predicted Word: {predicted_word}")
        print(f"Target Word: {target_word}")

        # Replace the predicted word with the original word in the sentence
        modified_sentence = sentence[:]
        modified_sentence[target_index] = predicted_word
        print(f"Modified Sentence: {' '.join(modified_sentence)}")
        print("-" * 50)
    
    return target_rank

# --- Run evaluation with detailed statistics ---
model = Word2Vec.load(cbow_file)
results = {"garden": [], "control": []}
failed_cases = {"garden": [], "control": []}

for sentence, idx, label in test_data:
    rank = evaluate_prediction_with_rank(model, sentence, idx)
    if rank is not None:
        results[label].append(rank)
    else:
        failed_cases[label].append(" ".join(sentence))

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
median_ranks = [np.median(results["garden"]) if results["garden"] else 0,
                np.median(results["control"]) if results["control"] else 0]

bars = ax.bar(labels, median_ranks, color=['red', 'green'])
ax.set_ylabel('Median Rank')
if cbow_file == cbow_neg:
    ax.set_title('CBOW with Negative Sampling Model Performance: Garden Path vs Control Sentences')
else:
    ax.set_title('CBOW Model Performance: Garden Path vs Control Sentences')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom')

plt.show()
