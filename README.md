# 21-366-FinalProject
This project explores the linguistic phenomenon of garden path sentences using a trained CBOW (Continuous Bag of Words) model. Garden path sentences are grammatically correct but lead the reader to initially interpret them incorrectly, creating ambiguity. The project compares the CBOW model's ability to process garden path sentences versus control sentences.

Workflow
  - Train the CBOW Model:
    Use the train_cbow.py script to train a CBOW model on the text8 dataset.
    The trained model is saved as cbow_text8.model.

  - Evaluate the Model:
    Use the case_study.py script to evaluate the CBOW model on the gardenpath_dataset.csv file.
    The dataset contains pairs of garden path and control sentences, along with the target word index.

  - Compare Results:
    The evaluation script computes the rank of the target word in the CBOW model's vocabulary based on its similarity to the context.
    Results include average and median ranks for garden path and control sentences, along with a bar chart visualization.

Dataset Example
The gardenpath_dataset.csv contains sentences like:
Sentence	| Target Index	| Label
  - The player tossed the ball interfered with the other team.	| 2	| garden
  - The player who was tossed the ball interfered with the other team.	| 4	| control

Key Features:
  - Training: Train a CBOW model on a large corpus (text8) for word embeddings. 
  - Evaluation: Analyze the model's ability to handle ambiguous sentences. 
  - Visualization: Compare performance on garden path vs. control sentences using statistical metrics and plots. 
