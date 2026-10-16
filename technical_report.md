
# Training Word2Vec and Emotion Classification Using GloVe

## 1. Introduction
This project studies custom Word2Vec embeddings and pretrained
GloVe representations for emotion recognition using GoEmotions.

The dataset contains 207805 usable comments and 28
emotion categories including neutral.

## 2. Word2Vec Training
Corpus source: GoEmotions comments (fallback)
Documents: 207805
Tokens: 2783484
Vocabulary after min_count=5: 19561

Skip-gram and CBOW were compared using different vector dimensions,
context windows, and epochs.

The experiment table is saved as word2vec_experiments.csv.

A probe similarity score is not a downstream accuracy measure.
WordSim-353 and SimLex-999 evaluation requires the respective
benchmark datasets.

## 3. GloVe Embeddings
GloVe 6B 100-dimensional pretrained embeddings were used.

GloVe vocabulary coverage: 94.33%
Word2Vec vocabulary coverage: 98.68%

Out-of-vocabulary words receive zero vectors.

## 4. Emotion Classification
The classification experiment used single-label examples.

Training, validation, and testing used a 70/15/15 split.

Neural architectures:
- Dense network with global average pooling
- CNN with global max pooling
- Fine-tuned GloVe Dense model
- Custom Word2Vec Dense model

Baselines:
- TF-IDF Logistic Regression
- TF-IDF Random Forest

Class weights were applied to single-label neural training.

## 5. Results
Best model by test macro F1: TFIDF_LogisticRegression
Test accuracy: 0.2908
Macro F1: 0.2864
Weighted F1: 0.2755

Full results:
                   Model  Accuracy  Macro_F1  Weighted_F1  Micro_F1
             GloVe_Dense  0.136156  0.106153     0.097113  0.136156
               GloVe_CNN  0.226214  0.225110     0.199450  0.226214
          Word2Vec_Dense  0.164985  0.153867     0.131066  0.164985
         GloVe_FineTuned  0.199596  0.213345     0.162561  0.199596
TFIDF_LogisticRegression  0.290781  0.286380     0.275517  0.290781
      TFIDF_RandomForest  0.344754  0.046435     0.191336  0.344754

## 6. Multi-label Emotion Recognition
Multi-label classification uses sigmoid outputs and
binary cross-entropy.

Metrics:
{
  "threshold": 0.2,
  "hamming_loss": 0.05291892189168504,
  "micro_f1": 0.406679940908215,
  "macro_f1": 0.2664902797061442,
  "weighted_f1": 0.34879223874454623,
  "precision_at_1": 0.4308812618255615,
  "recall_at_1": 0.3918308913707733,
  "precision_at_3": 0.24012702703475952,
  "recall_at_3": 0.6295592784881592,
  "precision_at_5": 0.17327646911144257,
  "recall_at_5": 0.7431163191795349
}

The decision threshold was selected on validation data.

## 7. Embedding Analysis
Word2Vec nearest-neighbor queries and analogy experiments
were performed where words existed in the vocabulary.

GloVe emotion words were visualized with t-SNE and PCA.
Emotion centroid similarity was computed using averaged
GloVe comment representations.

## 8. Error Analysis
Most confused pairs:
 Actual   Predicted  Count
neutral disapproval    666
neutral realization    535
neutral    approval    510
neutral   curiosity    508
neutral      caring    496
neutral   confusion    454
neutral      desire    331
neutral       anger    331
neutral  excitement    322
neutral    optimism    300

Highest per-class F1:
            precision    recall  f1-score
gratitude    0.844184  0.744821  0.791396
love         0.514286  0.726783  0.602342
amusement    0.500904  0.602174  0.546890
admiration   0.506867  0.490817  0.498713
remorse      0.290541  0.570796  0.385075
surprise     0.269675  0.493282  0.348711
fear         0.250000  0.545113  0.342790
curiosity    0.269590  0.425170  0.329960

Lowest per-class F1:
                precision    recall  f1-score
embarrassment    0.102310  0.288372  0.151035
neutral          0.664158  0.085122  0.150903
annoyance        0.191327  0.119904  0.147420
disappointment   0.135654  0.160057  0.146849
relief           0.085661  0.389831  0.140458
nervousness      0.072435  0.300000  0.116694
pride            0.048417  0.252427  0.081250
grief            0.046948  0.188679  0.075188

Error analysis includes comment length, lexical richness,
and emotion frequency.

## 9. Limitations
The GoEmotions comments are not an independently collected
domain-specific corpus.

Corpus minimum requirements:
10,000 documents: True
1 million tokens: True

The Word2Vec hyperparameter probe similarity is exploratory.
External WordSim-353 and SimLex-999 benchmarks were not run.

The code uses 100-dimensional GloVe; a separate 300-dimensional
experiment is not included.

The project compares two neural architectures, but does not
implement attention or a BiLSTM training experiment.

No minimum accuracy or F1 score is guaranteed.

## 10. Conclusion
The results provide an empirical comparison between sparse
TF-IDF features and dense Word2Vec/GloVe embeddings.

The strongest model is identified using held-out test macro F1.
For model selection in a stricter experimental protocol,
validation metrics should be used instead.

Future improvements include larger domain corpora, benchmark
similarity evaluation, 300d GloVe, and contextual embeddings.
