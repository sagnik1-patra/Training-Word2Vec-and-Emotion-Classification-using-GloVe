# Word2Vec and GloVe Emotion Classification

## Dataset
GoEmotions with 27 emotion categories plus neutral.

## Workflow
1. Load and clean data
2. Perform exploratory data analysis
3. Train Word2Vec
4. Compare Word2Vec hyperparameters
5. Load pretrained GloVe
6. Train emotion classifiers
7. Train TF-IDF baselines
8. Evaluate multi-label classification
9. Visualize embedding spaces
10. Analyze prediction errors

## Main Results
See `csv/model_comparison.csv` and
`reports/project_summary.json`.

## Visualizations
See the `graphs` directory.

## Limitations
See `reports/technical_report.md`.

## Reproduction
Install requirements and run the original Jupyter notebook.
The large input CSV and pretrained GloVe file are not included
in the compact archive.
