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
  - The gardenpath_dataset.csv contains pairs of garden path and control sentences such as:
```
  Sentence	                                                          | Target Index	| Label
  The player tossed the ball interfered with the other team.	        | 2	            | garden
  The player who was tossed the ball interfered with the other team.	| 4	            | control
```

## Workflow
  - Train the CBOW Model:
    Use the train_cbow.py script to train a CBOW model on the text8 dataset.
    The trained model is saved as cbow_text8.model.

  - Evaluate the Model:
    Use the case_study.py script to evaluate the CBOW model on the gardenpath_dataset.csv file.
    The dataset contains pairs of garden path and control sentences, along with the target word index.

  - Compare Results:
    The evaluation script computes the rank of the target word in the CBOW model's vocabulary based on its similarity to the context.
    Results include average and median ranks for garden path and control sentences, along with a bar chart visualization.
