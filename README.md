# Fake News Detection using DSA & Machine Learning

## Overview
This project focuses on detecting fake news (rumors) from Twitter threads using data structures and machine learning techniques. It utilizes the PHEME dataset to analyze conversation threads structured as trees.

## Dataset
- **PHEME dataset**: Includes events like charliehebdo, ebola, ferguson, etc.
- **Structure**: Data is organized as conversation threads in a tree-like format.

## Approach
1. Parse Twitter threads into tree structures using data structures.
2. Extract features from text content and structural properties (e.g., depth, breadth, user interactions).
3. Train a machine learning classification model to distinguish between real and fake news.

## Project Structure
```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # Raw dataset files
│   └── processed/    # Preprocessed data
├── notebooks/
│   └── exploration.ipynb  # Jupyter notebook for data exploration
├── results/
│   ├── metrics.txt   # Model evaluation metrics
│   └── figures/      # Plots and visualizations
└── src/
    ├── preprocessing.py     # Data loading and cleaning
    ├── feature_engineering.py  # Feature extraction
    ├── model.py             # Model training and evaluation
    └── utils.py             # Utility functions
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fake-news-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the PHEME dataset and place it in `data/raw/`.

## Usage
Run the entire pipeline:
```bash
python run.py
```

Or run individual steps:
1. Preprocess the data:
   ```bash
   python src/preprocessing.py
   ```

2. Extract features:
   ```bash
   python src/feature_engineering.py
   ```

3. Train the model:
   ```bash
   python src/model.py
   ```

4. View results in `results/metrics.txt` and `results/figures/`.

## Results
- **Accuracy**: 0.8500
- **Precision**: 0.8333
- **Recall**: 0.8571
- **F1-Score**: 0.8451

## Dependencies
- numpy
- pandas
- scikit-learn
- nltk
- matplotlib

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
[Add contribution guidelines]
