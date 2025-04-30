# 21-366-FinalProject
This project evaluates the performance of the Continuous Bag-of-Words (CBOW) word embedding model on garden path sentences. It explores how adding negative sampling and tuning hyperparameters like window size affects performance on syntactically ambiguous text.


## Project Structure

```
.
├── data/                      # Dataset
│   ├── gardenpath_dataset.csv # Custom dataset of garden path/control sentence pairs
│
├── train/                    # Contains training scripts for CBOW models
│   ├── train_cbow.py         # Trains baseline CBOW model on Text8 corpus
│   ├── train_neg_cbow.py     # Trains CBOW with negative sampling + wider window
│
├── models/                   # Stores trained models
│   ├── cbow_text8.model      # Baseline model
│   ├── cbow_neg.model        # Model with neg sampling
│
└── run_case_study.py         #  Loads models and dataset, evaluates, and outputs results
```


## Features
- **Training**: Train a CBOW model on a large corpus (text8) for word embeddings. 
- **Evaluation**: Analyze the model's ability to handle ambiguous sentences. 
- **Visualization**: Compare performance on garden path vs. control sentences using statistical metrics and plots. 


## Dataset Example
The gardenpath_dataset.csv contains pairs of garden path and control sentences such as:
```
  Sentence	                        | Target Index	 | Label
  "The old man the boats"	        | 2	            | garden
  "The old people man the boats"	| 3	            | control
```
- **sentence**: full sentence as a string
- **target_index**: index of the syntactically important word to predict
- **label**: either garden or control


## Workflow
**Model Training**:
- Trained two CBOW models on the Wikipedia Text8 corpus using Gensim Word2Vec.
- Baseline model: window size = 3, embedding size = 100, epochs = 5, no negative sampling.
- Optimized model: negative sampling with 20 samples, window size = 5.

**Evaluation Setup**:
- Used a custom dataset of garden path and control sentence pairs.
- For each sentence, masked a syntactically important word (e.g., verb), predicted it from context.
- Computed cosine similarity between averaged context vector and all vocabulary words.
- Evaluated performance using the rank of the correct word: lower rank = better prediction.

## Results
**Figure 1** shows that the baseline CBOW model performs significantly worse on garden path sentences than on control sentences, with higher median ranks indicating difficulty in resolving syntactic ambiguity.

**Figure 2** demonstrates that the optimized CBOW model (with negative sampling and a wider context window) substantially improves performance on garden path sentences — reducing median rank by ~50% — while maintaining similar performance on control sentences.
<img width="360" alt="image" src="https://github.com/user-attachments/assets/873d1cf2-379f-4c68-9b97-3ccb6c3aef90" />
<img width="360" alt="image" src="https://github.com/user-attachments/assets/238b642d-c3c5-492a-b19d-b8e96fcb1516" />


