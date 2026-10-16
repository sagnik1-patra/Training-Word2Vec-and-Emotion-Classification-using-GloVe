
# Training Word2Vec and Emotion Classification Using GloVe

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-green)
![NLP](https://img.shields.io/badge/NLP-Emotion%20Classification-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

**Training Word2Vec and Emotion Classification Using GloVe** is a Natural Language Processing (NLP) and Deep Learning project designed to explore word embeddings and automatically recognize emotions expressed in textual data.

The project uses the **GoEmotions dataset**, consisting of Reddit comments annotated with 27 emotion categories and a neutral category.

It combines custom-trained **Word2Vec embeddings**, pretrained **GloVe embeddings**, neural network classifiers, and traditional machine learning algorithms to compare different approaches to emotion recognition.

The project includes:

- Custom Word2Vec training using Skip-gram.
- Word2Vec hyperparameter experiments.
- Pretrained GloVe word embeddings.
- Dense Neural Network emotion classification.
- Convolutional Neural Network (CNN) classification.
- Word2Vec-based emotion classification.
- Fine-tuned GloVe classification.
- TF-IDF with Logistic Regression.
- TF-IDF with Random Forest.
- Multi-label emotion recognition.
- Embedding visualization using t-SNE and PCA.
- Model performance comparison.
- Error analysis and confusion matrices.
- Automated result generation and ZIP packaging.

The objective is to investigate how dense word representations compare with traditional sparse text representations in emotion classification.

---

## 📊 Dataset Visualization

### Emotion Class Distribution

The following visualization displays the frequency distribution of emotion labels in the GoEmotions dataset.

![GoEmotions Class Distribution](graphs/01_class_distribution.png)

**Figure 1:** Distribution of emotion labels across the GoEmotions dataset.

This visualization helps identify:

- Frequently occurring emotions.
- Underrepresented emotion categories.
- Class imbalance within the dataset.
- Differences in emotion label frequencies.
- Potential challenges when training classification models.

Class imbalance is important because models may perform better on frequently occurring emotions than on rare categories.

---

## 🎯 Project Objectives

The primary objectives of this project are:

1. Develop a custom Word2Vec embedding model.
2. Understand semantic relationships between words.
3. Compare Skip-gram and CBOW architectures.
4. Investigate Word2Vec hyperparameters.
5. Integrate pretrained GloVe embeddings.
6. Build deep learning models for emotion classification.
7. Compare Word2Vec and GloVe representations.
8. Implement traditional TF-IDF classification baselines.
9. Perform single-label and multi-label emotion classification.
10. Evaluate models using classification metrics.
11. Visualize word embeddings and emotion relationships.
12. Analyze classification errors and model limitations.
13. Automatically generate graphs, reports, and prediction files.
14. Package the results into a compressed ZIP archive under 50 MB.

---

## 📂 Dataset Description

### GoEmotions Dataset

**Dataset Name:** GoEmotions

**Source:** Google Research

**Dataset File:** `go_emotions_dataset.csv`

**Original Dataset:** Approximately 58,000 Reddit comments in the simplified GoEmotions release.

**Number of Emotion Categories:** 27 emotions + neutral.

**Dataset Type:** Multi-label text classification.

**Language:** English.

### Emotion Categories

The dataset includes the following labels:

| No. | Emotion | No. | Emotion |
|---|---|---|---|
| 1 | Admiration | 15 | Gratitude |
| 2 | Amusement | 16 | Grief |
| 3 | Anger | 17 | Joy |
| 4 | Annoyance | 18 | Love |
| 5 | Approval | 19 | Nervousness |
| 6 | Caring | 20 | Optimism |
| 7 | Confusion | 21 | Pride |
| 8 | Curiosity | 22 | Realization |
| 9 | Desire | 23 | Relief |
| 10 | Disappointment | 24 | Remorse |
| 11 | Disapproval | 25 | Sadness |
| 12 | Disgust | 26 | Surprise |
| 13 | Embarrassment | 27 | Neutral |
| 14 | Excitement | | |
| 28 | Fear | | |

### Dataset Features

The CSV file contains:

- `text`: Reddit comment text.
- `example_very_unclear`: Indicator for ambiguous annotations.
- Emotion columns: Binary indicators representing emotion labels.

Each comment can have one or multiple emotion labels.

### Example

| Text | Joy | Love | Sadness |
|---|---|---|---|
| I am feeling wonderful today | 1 | 0 | 0 |
| I love spending time with you | 0 | 1 | 0 |
| I feel extremely disappointed | 0 | 0 | 1 |

*The table contains illustrative examples, not verified rows from the dataset.*

---

## 🏗️ System Architecture

The complete workflow consists of the following stages:

```text
                   GoEmotions Dataset
                           |
                           v
                    Data Preprocessing
                           |
             +-------------+-------------+
             |                           |
             v                           v
      Word2Vec Training           GloVe Embeddings
             |                           |
      Skip-gram / CBOW             Embedding Matrix
             |                           |
      Word Similarity              Neural Networks
             |                           |
      Embedding Analysis           Dense / CNN
             |                           |
             +-------------+-------------+
                           |
                           v
                   Emotion Prediction
                           |
                           v
                    Model Evaluation
                           |
                           v
                Performance Comparison
                           |
                           v
                   Error Analysis
                           |
                           v
                Graphs, CSVs and Reports
                           |
                           v
                   Final ZIP Archive
```

---

## 🛠️ Technologies Used

### Programming Language

- Python 3.10+

### Machine Learning

- Scikit-learn
- Logistic Regression
- Random Forest
- TF-IDF Vectorization

### Deep Learning

- TensorFlow
- Keras
- Dense Neural Networks
- Convolutional Neural Networks
- Embedding Layers

### NLP

- Gensim
- Word2Vec
- GloVe
- Tokenization
- Text Cleaning

### Data Processing

- NumPy
- Pandas
- SciPy

### Visualization

- Matplotlib
- Seaborn
- t-SNE
- PCA

### File Management

- JSON
- YAML
- Joblib
- HDF5
- ZIP compression

---

## 🔄 Project Methodology

### Phase 1: Data Preprocessing

The first stage prepares the raw comments for NLP analysis.

Preprocessing operations include:

1. Loading the CSV dataset.
2. Removing missing text entries.
3. Removing empty comments.
4. Filtering unclear examples when indicated.
5. Converting text to lowercase.
6. Removing URLs.
7. Removing unwanted special characters.
8. Tokenizing sentences into words.
9. Preserving meaningful hyphenated words.
10. Removing examples without emotion labels.

Example:

**Original Text:**

```text
I am REALLY happy today!!! Visit https://example.com
```

**Processed Text:**

```text
i am really happy today visit
```

The preprocessing process prepares consistent textual input for word embedding and classification models.

---

## 🧠 Phase 2: Custom Word2Vec Training

Word2Vec is a neural embedding technique that learns vector representations of words based on their surrounding context.

Words occurring in similar contexts tend to develop similar vector representations.

### Word2Vec Architectures

**1. Continuous Bag of Words (CBOW)**

CBOW predicts a target word from its surrounding context words.

```text
Context Words
      |
      v
Embedding Layer
      |
      v
Hidden Representation
      |
      v
Target Word
```

**2. Skip-gram**

Skip-gram predicts surrounding context words using a target word.

```text
Target Word
      |
      v
Embedding Layer
      |
      v
Hidden Representation
      |
      v
Context Words
```

### Word2Vec Configuration

| Parameter | Value |
|---|---|
| Vector Size | 100 |
| Window Size | 5 |
| Minimum Word Frequency | 5 |
| Architecture | Skip-gram |
| Training Epochs | 10 |
| Negative Samples | 5 |
| Workers | 4 |

### Hyperparameter Experiments

The project investigates:

- Embedding dimensions: 50, 100, 200, 300.
- Context windows: 2, 5, 10.
- Architectures: Skip-gram and CBOW.
- Training epochs: 5, 10, 20.

The experiments compare vocabulary size and average similarity across selected semantic word pairs.

### Saved Word2Vec Model

```text
models/word2vec_custom.model
```

### Word Similarity Analysis

The model investigates semantic similarity between related words.

Example:

```python
model.wv.most_similar("happy", topn=10)
```

Additional operations include:

```python
model.wv.similarity("happy", "joy")
```

### Word Analogy Analysis

The project also evaluates vector arithmetic.

Example:

```text
king - man + woman ≈ queen
```

Emotion-related analogies include:

```text
happy - joy + sadness
love - like + hate
```

These tasks investigate whether the embeddings capture meaningful semantic relationships.

---

## 🌐 Phase 3: GloVe Embeddings

GloVe stands for **Global Vectors for Word Representation**.

Unlike Word2Vec, which learns from local context prediction, GloVe learns word representations using global word co-occurrence statistics.

### Pretrained Embedding Configuration

| Parameter | Value |
|---|---|
| Embedding Type | GloVe |
| Corpus | Wikipedia + Gigaword |
| Embedding Dimensions | 100 |
| Source | Stanford NLP |
| File | glove.6B.100d.txt |

### GloVe Download

Official download:

https://nlp.stanford.edu/data/glove.6B.zip

### Embedding Matrix

A vocabulary is generated from the training dataset.

Each word is assigned a corresponding GloVe vector when available.

For out-of-vocabulary words, the implementation uses zero vectors.

The embedding matrix has the following shape:

```text
Vocabulary Size × Embedding Dimension
```

Example:

```text
15000 × 100
```

### Embedding Coverage

Embedding coverage measures the percentage of vocabulary words with available pretrained vectors.

```text
Coverage (%) =
(Matched Vocabulary Words / Vocabulary Words) × 100
```

The project compares coverage between:

- Custom Word2Vec.
- Pretrained GloVe.

---

## 🤖 Phase 4: Emotion Classification Models

The project implements several classification models.

### Model 1: GloVe + Dense Neural Network

Architecture:

```text
Input Text
    |
Tokenization
    |
Padding
    |
GloVe Embedding
    |
Global Average Pooling
    |
Dense (128, ReLU)
    |
Dropout (0.5)
    |
Dense (64, ReLU)
    |
Dropout (0.3)
    |
Softmax Output
```

This model generates fixed-size text representations by averaging word embeddings.

### Model 2: GloVe + CNN

Architecture:

```text
Input Text
    |
GloVe Embedding
    |
Conv1D (128 filters)
    |
Global Max Pooling
    |
Dense (128, ReLU)
    |
Dropout (0.5)
    |
Softmax Output
```

CNNs capture local patterns in word sequences.

They can identify phrases and word combinations associated with different emotions.

### Model 3: Word2Vec + Dense Neural Network

This architecture uses custom-trained Word2Vec embeddings instead of pretrained GloVe vectors.

It allows comparison between:

- Corpus-trained embeddings.
- General-purpose pretrained embeddings.

### Model 4: Fine-Tuned GloVe Network

The embedding layer is made trainable.

This allows pretrained GloVe vectors to be updated during emotion classification training.

The experiment investigates whether fine-tuning improves performance compared with frozen embeddings.

---

## 📈 Phase 5: Traditional Machine Learning Baselines

Two traditional machine learning approaches are implemented.

### TF-IDF + Logistic Regression

TF-IDF transforms text into sparse numerical feature vectors.

Logistic Regression learns the relationship between these features and emotion categories.

### TF-IDF + Random Forest

Random Forest combines multiple decision trees to perform classification.

The model uses TF-IDF features as input.

### Baseline Comparison

The project compares:

| Model | Feature Representation |
|---|---|
| Logistic Regression | TF-IDF |
| Random Forest | TF-IDF |
| Word2Vec Dense | Custom Word2Vec |
| GloVe Dense | Pretrained GloVe |
| GloVe CNN | Pretrained GloVe |
| GloVe Fine-Tuned | Trainable GloVe |

The comparison helps determine whether dense word embeddings provide measurable improvements over sparse text features.

---

## 🏷️ Phase 6: Multi-Label Emotion Classification

GoEmotions allows a single comment to express multiple emotions.

For example:

```text
I am happy and excited about the surprise!
```

Possible labels:

```text
Joy = 1
Excitement = 1
Surprise = 1
```

### Multi-Label Architecture

```text
Input Text
    |
GloVe Embedding
    |
Conv1D
    |
Global Max Pooling
    |
Dense Layer
    |
Dropout
    |
Sigmoid Output
```

### Loss Function

Binary Cross-Entropy:

```text
Loss = -[y log(p) + (1-y) log(1-p)]
```

Unlike softmax, sigmoid allows multiple emotion labels to be predicted independently.

### Evaluation Metrics

The model is evaluated using:

- Hamming Loss.
- Micro F1-score.
- Macro F1-score.
- Weighted F1-score.
- Precision@1, Precision@3, Precision@5.
- Recall@1, Recall@3, Recall@5.

The classification threshold is selected using validation data.

---

## 📊 Phase 7: Model Evaluation

The project evaluates models using several performance metrics.

### Accuracy

Accuracy measures the proportion of correctly classified examples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1-Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Macro F1

Macro F1 calculates the average F1-score across emotion categories.

Each category receives equal importance.

### Weighted F1

Weighted F1 calculates the average F1-score weighted by the number of examples belonging to each category.

### Confusion Matrix

The confusion matrix shows how frequently one emotion is predicted as another.

This helps identify difficult emotion pairs.

### Model Comparison Results

Actual model scores are automatically generated during training.

The results are saved in:

```text
csv/model_comparison.csv
```

The project does not assume a predefined winning model. Performance depends on the dataset, training conditions, and learned representations.

---

## 📉 Phase 8: Embedding Visualization

The project investigates the geometric properties of word embeddings.

### t-SNE Visualization

t-SNE reduces high-dimensional word vectors to two dimensions.

The visualization helps explore relationships among words in the embedding space.

### Emotion Word Groups

**Happy:**

```text
joy, happy, excited, thrilled
```

**Sad:**

```text
sad, depressed, unhappy, miserable
```

**Anger:**

```text
angry, furious, mad, rage
```

**Love:**

```text
love, affection, romance, adoration
```

**Fear:**

```text
fear, scared, afraid, terrified
```

### PCA Visualization

Principal Component Analysis is used to project emotion embeddings into a three-dimensional space.

### Emotion Centroids

An emotion centroid represents the average embedding associated with an emotion category.

Centroid analysis helps explore relationships between emotion classes.

### Emotion Similarity Matrix

Cosine similarity measures the similarity between emotion centroid vectors.

```text
Cosine Similarity =
(A · B) / (||A|| × ||B||)
```

---

## 🔍 Phase 9: Error Analysis

Error analysis investigates why classification models produce incorrect predictions.

### Confusion Analysis

The project identifies frequently confused emotion categories.

Examples of potentially overlapping emotions include:

- Joy and excitement.
- Anger and annoyance.
- Sadness and disappointment.
- Approval and admiration.
- Fear and nervousness.

These are illustrative possibilities; actual confusion frequencies are calculated from the model predictions.

### Comment Length Analysis

The project examines whether classification accuracy changes with comment length.

### Emotion Frequency Analysis

The project compares classification performance for common and rare emotions.

### Lexical Complexity Analysis

Lexical richness is calculated using:

```text
Lexical Richness =
Unique Words / Total Words
```

This helps investigate how vocabulary diversity relates to classification performance.

### Misclassified Examples

Incorrect predictions are saved for inspection.

```text
csv/misclassified_examples.csv
```

---

## 📁 Project Folder Structure

```text
Training Word2Vec and Emotion Classification using GloVe/
│
├── go_emotions_dataset.csv
│
├── assets/
│   └── glove.6B.100d.txt
│
├── results/
│   │
│   ├── graphs/
│   │   ├── 01_class_distribution.png
│   │   ├── 02_comment_length.png
│   │   ├── 03_multilabel_distribution.png
│   │   ├── 04_word2vec_parameter_comparison.png
│   │   ├── 05_word2vec_tsne.png
│   │   ├── 06_embedding_coverage.png
│   │   ├── 07_model_accuracy_comparison.png
│   │   ├── 08_model_f1_comparison.png
│   │   ├── 09_confusion_matrix.png
│   │   ├── 10_per_class_f1.png
│   │   ├── 11_confused_emotion_pairs.png
│   │   ├── 12_accuracy_by_length.png
│   │   ├── 13_glove_emotion_tsne.png
│   │   ├── 14_emotion_embedding_3d.png
│   │   ├── 15_emotion_centroids.png
│   │   ├── 16_emotion_similarity_matrix.png
│   │   ├── 17_multilabel_training_loss.png
│   │   ├── 18_frequency_vs_f1.png
│   │   └── 19_lexical_complexity.png
│   │
│   ├── models/
│   │   ├── word2vec_custom.model
│   │   ├── glove_embedding_matrix.npy
│   │   ├── word2vec_embedding_matrix.npy
│   │   ├── GloVe_Dense.h5
│   │   ├── GloVe_CNN.h5
│   │   ├── Word2Vec_Dense.h5
│   │   ├── GloVe_FineTuned.h5
│   │   ├── multilabel_emotion_classifier.h5
│   │   ├── logistic_regression.pkl
│   │   ├── random_forest.pkl
│   │   └── tfidf_vectorizer.pkl
│   │
│   ├── csv/
│   │   ├── dataset_preview.csv
│   │   ├── emotion_frequencies.csv
│   │   ├── emotion_examples.csv
│   │   ├── word2vec_experiments.csv
│   │   ├── word2vec_similarities.csv
│   │   ├── word2vec_analogies.csv
│   │   ├── model_comparison.csv
│   │   ├── classification_report.csv
│   │   ├── misclassified_examples.csv
│   │   ├── confused_emotion_pairs.csv
│   │   ├── emotion_similarity_matrix.csv
│   │   ├── emotion_frequency_vs_f1.csv
│   │   ├── multilabel_predictions.csv
│   │   └── *_predictions.csv
│   │
│   ├── reports/
│   │   ├── technical_report.md
│   │   ├── project_summary.json
│   │   ├── corpus_statistics.json
│   │   ├── embedding_coverage.json
│   │   ├── multilabel_metrics.json
│   │   ├── configuration.yaml
│   │   ├── tokenizer_vocabulary.json
│   │   ├── multilabel_vocabulary.json
│   │   ├── word2idx.json
│   │   ├── idx2word.json
│   │   └── zip_manifest.json
│   │
│   ├── README.md
│   └── requirements.txt
│
└── Word2Vec_GloVe_Final_Results.zip
```

Additional training-history graphs are generated for the individual neural network models.

---

## ⚙️ Installation and Setup

### Step 1: Install Python

Install Python 3.10 or a compatible version supported by the required TensorFlow release.

Download:

https://www.python.org/downloads/

### Step 2: Install Dependencies

Open a terminal or Jupyter Notebook and install:

```bash
pip install numpy pandas scipy scikit-learn gensim tensorflow matplotlib seaborn nltk requests tqdm joblib pyyaml h5py
```

Alternatively, use:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset

Place the dataset in:

```text
C:\Users\sagni\Downloads\Training Word2Vec and Emotion Classification using GloVe\go_emotions_dataset.csv
```

### Step 4: Run the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and execute the main workflow cell.

The first execution may download the pretrained GloVe embeddings.

An internet connection is required for this download.

### Step 5: Generate Results

The notebook automatically:

1. Loads the dataset.
2. Preprocesses the comments.
3. Trains Word2Vec models.
4. Loads pretrained GloVe embeddings.
5. Trains emotion classification models.
6. Evaluates classification performance.
7. Generates graphs and prediction files.
8. Performs embedding analysis.
9. Generates technical reports.
10. Creates the final ZIP archive.

---

## 💾 Generated Model Files

### Word2Vec Model

```text
models/word2vec_custom.model
```

Contains the custom-trained Word2Vec model.

### GloVe Dense Model

```text
models/GloVe_Dense.h5
```

Contains the trained Dense Neural Network using pretrained GloVe embeddings.

### GloVe CNN Model

```text
models/GloVe_CNN.h5
```

Contains the trained convolutional emotion classifier.

### Word2Vec Dense Model

```text
models/Word2Vec_Dense.h5
```

Contains the emotion classifier trained with custom Word2Vec embeddings.

### Fine-Tuned GloVe Model

```text
models/GloVe_FineTuned.h5
```

Contains the Dense classifier with trainable GloVe embeddings.

### Multi-Label Model

```text
models/multilabel_emotion_classifier.h5
```

Contains the multi-label emotion classification model.

### Machine Learning Models

```text
models/logistic_regression.pkl
models/random_forest.pkl
models/tfidf_vectorizer.pkl
```

These files store the traditional machine learning models and TF-IDF vectorizer.

---

## 📊 Generated Visualizations

The project generates the following main graphs:

| No. | Visualization | File |
|---|---|---|
| 1 | Emotion Class Distribution | `01_class_distribution.png` |
| 2 | Comment Length Distribution | `02_comment_length.png` |
| 3 | Multi-label Distribution | `03_multilabel_distribution.png` |
| 4 | Word2Vec Parameter Comparison | `04_word2vec_parameter_comparison.png` |
| 5 | Word2Vec t-SNE | `05_word2vec_tsne.png` |
| 6 | Embedding Coverage | `06_embedding_coverage.png` |
| 7 | Model Accuracy Comparison | `07_model_accuracy_comparison.png` |
| 8 | Model F1 Comparison | `08_model_f1_comparison.png` |
| 9 | Confusion Matrix | `09_confusion_matrix.png` |
| 10 | Per-Class F1 | `10_per_class_f1.png` |
| 11 | Confused Emotion Pairs | `11_confused_emotion_pairs.png` |
| 12 | Accuracy by Comment Length | `12_accuracy_by_length.png` |
| 13 | GloVe Emotion t-SNE | `13_glove_emotion_tsne.png` |
| 14 | 3D Emotion Embeddings | `14_emotion_embedding_3d.png` |
| 15 | Emotion Centroids | `15_emotion_centroids.png` |
| 16 | Emotion Similarity Matrix | `16_emotion_similarity_matrix.png` |
| 17 | Multi-label Training Loss | `17_multilabel_training_loss.png` |
| 18 | Frequency vs F1 | `18_frequency_vs_f1.png` |
| 19 | Lexical Complexity | `19_lexical_complexity.png` |

The notebook displays these figures and saves them as PNG files.

---

## 📋 Results and Prediction Files

### Model Comparison

```text
csv/model_comparison.csv
```

Contains model accuracy and F1-score results.

### Classification Report

```text
csv/classification_report.csv
```

Contains per-class precision, recall, F1-score, and support.

### Single-Label Predictions

```text
csv/GloVe_Dense_predictions.csv
csv/GloVe_CNN_predictions.csv
csv/Word2Vec_Dense_predictions.csv
csv/GloVe_FineTuned_predictions.csv
```

These files contain:

- Input text.
- Actual emotion.
- Predicted emotion.
- Prediction confidence.

### Multi-Label Predictions

```text
csv/multilabel_predictions.csv
```

Contains actual and predicted emotion combinations.

### Error Analysis

```text
csv/misclassified_examples.csv
csv/confused_emotion_pairs.csv
```

These files support detailed examination of incorrect predictions.

---

## 📦 Final ZIP Archive

The project generates:

```text
Word2Vec_GloVe_Final_Results.zip
```

### ZIP Size Requirement

The final archive is constrained to remain below 50 MB.

The packaging process prioritizes:

- Graphs.
- CSV results.
- Evaluation reports.
- Configuration files.
- Documentation.
- Compact model files.

Large pretrained embedding files and the original dataset are excluded from the compressed deliverables archive.

If necessary, larger trained model files are also excluded from the ZIP but remain available in the local `results/models` folder.

The ZIP is therefore a compact results package rather than a guaranteed fully standalone reproduction environment.

---

## 📚 Technical Report

The generated technical report is stored at:

```text
reports/technical_report.md
```

It covers:

1. Project introduction.
2. Word2Vec training methodology.
3. Hyperparameter experiments.
4. GloVe embedding analysis.
5. Emotion classification architectures.
6. Model performance comparison.
7. Multi-label classification.
8. Embedding visualization.
9. Classification error analysis.
10. Limitations and future improvements.

The report uses the actual metrics calculated during execution.

---

## 🔬 Experimental Considerations

### Word2Vec vs GloVe

Custom Word2Vec embeddings learn from the available training corpus.

Pretrained GloVe embeddings contain statistical information learned from a much larger general-purpose corpus.

Their relative performance depends on vocabulary coverage, domain similarity, model architecture, and training data.

### Frozen vs Fine-Tuned Embeddings

Frozen embeddings retain their pretrained values.

Fine-tuned embeddings update their values during supervised training.

Fine-tuning may improve domain adaptation but can increase the risk of overfitting.

### Dense vs CNN

Dense models provide a comparatively simple architecture.

CNN models can capture local word sequence patterns.

Their performance is compared empirically using the generated evaluation results.

### Class Imbalance

Rare emotion categories may have fewer training examples.

Class weighting is used in the single-label neural experiments to address this issue.

Macro F1 is particularly useful for assessing performance across imbalanced emotion categories.

---

## ⚠️ Limitations

The project has several limitations:

1. The supplied GoEmotions dataset may not satisfy the assignment's independent domain-corpus requirement.

2. Word2Vec training quality depends on corpus size and vocabulary coverage.

3. GloVe experiments use 100-dimensional embeddings; 300-dimensional experiments require additional implementation.

4. WordSim-353 and SimLex-999 intrinsic benchmarks are not implemented in the supplied notebook.

5. The neural architecture comparison includes Dense and CNN models, but does not train a BiLSTM or attention model.

6. Emotion labels can overlap semantically, making classification challenging.

7. Class imbalance can reduce performance on rare emotions.

8. Zero-vector handling of out-of-vocabulary words may discard useful information.

9. The final ZIP may exclude large trained models to satisfy the archive size constraint.

10. Achieving a particular accuracy or F1-score is not guaranteed.

---

## 🚀 Future Enhancements

Potential improvements include:

- Training Word2Vec on a larger independent domain-specific corpus.
- Using 300-dimensional GloVe embeddings.
- Implementing Bidirectional LSTM architectures.
- Adding attention-based neural networks.
- Integrating BERT or RoBERTa embeddings.
- Performing WordSim-353 and SimLex-999 evaluations.
- Implementing more extensive hyperparameter tuning.
- Applying advanced multi-label threshold optimization.
- Improving classification of minority emotion categories.
- Building an interactive emotion prediction dashboard.
- Deploying the model using Streamlit or Flask.
- Providing real-time emotion analysis through a web application.

---

## 🌍 Applications

### Social Media Sentiment Analysis

Analyze emotional responses in online discussions.

### Customer Feedback Analysis

Identify emotions expressed in product reviews and customer support messages.

### Conversational AI

Help conversational systems recognize emotional cues in text.

### Market Research

Study emotional reactions to brands, products, and campaigns.

### Educational Feedback Analysis

Analyze emotions expressed in student feedback and survey responses.

### Online Community Research

Investigate emotional patterns in large-scale text discussions.

---

## 📖 References

### 1. GoEmotions Dataset

Demszky, D., et al. (2020).

*GoEmotions: A Dataset of Fine-Grained Emotions.*

ACL 2020.

https://github.com/google-research/google-research/tree/master/goemotions

### 2. Word2Vec

Mikolov, T., et al. (2013).

*Efficient Estimation of Word Representations in Vector Space.*

https://arxiv.org/abs/1301.3781

### 3. GloVe

Pennington, J., Socher, R., and Manning, C. (2014).

*GloVe: Global Vectors for Word Representation.*

https://nlp.stanford.edu/projects/glove/

### 4. Gensim

https://radimrehurek.com/gensim/

### 5. TensorFlow

https://www.tensorflow.org/

### 6. Scikit-learn

https://scikit-learn.org/

---

## 🏁 Conclusion

This project presents a comprehensive experimental workflow for training and evaluating word embeddings for emotion recognition.

By integrating **Word2Vec, GloVe, Dense Neural Networks, CNNs, Logistic Regression, and Random Forest**, the system provides a framework for comparing sparse and dense text representations.

The project also includes multi-label emotion recognition, semantic embedding analysis, performance visualization, and error analysis.

The generated results provide evidence for assessing the effectiveness and limitations of different NLP techniques on fine-grained emotion classification.

---

## 📄 License

This project is intended for academic, educational, and research purposes.

The GoEmotions dataset, pretrained GloVe embeddings, and third-party libraries remain subject to their respective licenses and usage terms.

---

## ⭐ Project Summary

**Project:** Training Word2Vec and Emotion Classification Using GloVe

**Domain:** Natural Language Processing and Deep Learning

**Dataset:** GoEmotions

**Classes:** 27 Emotions + Neutral

**Embeddings:** Word2Vec and GloVe

**Models:** Dense, CNN, Logistic Regression, Random Forest

**Tasks:** Single-label and Multi-label Emotion Classification

**Visualizations:** 19+ graphs

**Output Formats:** PNG, CSV, JSON, YAML, H5, PKL, MODEL, NPY, ZIP

**Final Deliverable:** Compressed Results Archive Under 50 MB

---

**Developed as an NLP and Deep Learning project for exploring semantic word representations and fine-grained emotion recognition.**
