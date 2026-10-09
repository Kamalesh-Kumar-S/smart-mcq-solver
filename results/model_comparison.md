# Model Comparison

The project explored several approaches for ranking MCQ answer choices.

| Model | Approach | Development result |
|---|---|---:|
| TF-IDF + PyTorch NN | TF-IDF features followed by a feed-forward neural network | 100% training accuracy by epoch 2 |
| all-MiniLM-L6-v2 | Prompt and option embeddings + cosine similarity | 26.1% training top-1 accuracy |
| all-mpnet-base-v2 | Prompt and option embeddings + cosine similarity | 25.9% training top-1 accuracy |

The official competition metric was **MAP@3**. The values above are development diagnostics recorded during experimentation and should not be interpreted as official Kaggle leaderboard scores.
