# Bioinformatics Analysis Pipeline

A comprehensive Python-based pipeline for analyzing genomic sequences using advanced feature extraction techniques and machine learning algorithms.

## Overview

This pipeline provides a complete workflow for processing genomic sequences, extracting meaningful features, and applying various machine learning models to identify patterns or mutations. It's designed to be flexible, extensible, and suitable for various bioinformatics research applications.

## Features

- **Multiple Feature Extraction Methods**:
  - K-mer frequency analysis
  - Amino acid composition
  - Physicochemical properties (hydrophobicity)

- **Advanced Machine Learning Models**:
  - Random Forest with hyperparameter tuning
  - Gradient Boosting Machine (GBM)
  - Support Vector Machine (SVM)
  - XGBoost
  - Neural Networks (Tensorflow/Keras)

- **Pipeline Components**:
  - Sequence loading and preprocessing
  - Feature normalization
  - Model training and evaluation
  - Cross-validation
  - Feature importance visualization
  - Model saving for future use

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Dependencies

Install all required packages using pip:

```bash
pip install biopython numpy pandas scikit-learn matplotlib xgboost tensorflow scipy
```

Or install directly from the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/
```

This command runs the pipeline with default settings (Random Forest model, k=3 for k-mers, 20% test split).

### Advanced Usage

#### Specify the Machine Learning Model

```bash
# Use Gradient Boosting Machine
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --model gbm

# Use Neural Network
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --model nn

# Use Support Vector Machine
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --model svm

# Use XGBoost
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --model xgb

# Try all available models
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --model all
```

#### Customize Feature Extraction

```bash
# Use a different k-mer size
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --k 4

# Use a different test set size
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --test_size 0.3
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input FASTA file | Required |
| `--output` | Output directory for results | `results` |
| `--test_size` | Test set size (proportion) | `0.2` |
| `--k` | K-mer size for feature extraction | `3` |
| `--model` | Machine learning model to use (rf, gbm, svm, xgb, nn, all) | `rf` |

## Input Data Format

The pipeline accepts genomic sequences in FASTA format:

```
>sequence_id1|optional_metadata
ATGCAGTACGTAGCTAGCTAGCTAGCTACGATCGATCG
>sequence_id2|optional_metadata
ATGCAGTACGTAGCTAGCCAGCTACGATCGATCGATCG
```

## Output

The pipeline generates the following outputs in the specified directory:

- **Model Files**: Saved trained models for future use
  - `rf_model.pkl` (Random Forest)
  - `gbm_model.pkl` (Gradient Boosting Machine)
  - `svm_model.pkl` (Support Vector Machine)
  - `xgb_model.pkl` (XGBoost)
  - `nn_model/` (Neural Network directory)

- **Normalization Parameters**:
  - `feature_scaler.pkl`: StandardScaler parameters for feature normalization

- **Visualizations**:
  - Feature importance plots for tree-based models
    - `rf_feature_importance.png`
    - `gbm_feature_importance.png`
    - `xgb_feature_importance.png`

## Extending the Pipeline

### Adding New Feature Extraction Methods

To add a new feature extraction method, implement a function in the script and update the `extract_features` function:

```python
def my_new_feature_method(sequence):
    # Implementation
    return features

# Then in extract_features():
# Add your new features
new_features = [my_new_feature_method(seq) for seq in sequences]
new_features_df = pd.DataFrame(new_features)
# Combine with existing features
combined_features = pd.concat([existing_features_df, new_features_df], axis=1)
```

### Adding New Machine Learning Models

To add a new model:

1. Implement a training function:

```python
def train_new_model(X_train, y_train):
    # Initialize and train your model
    return trained_model
```

2. Update the model selection in the `main()` function:

```python
if model_name == 'new_model':
    model = train_new_model(X_train, y_train)
```

3. Update the available models in `parse_arguments()`:

```python
parser.add_argument('--model', type=str, default='rf', 
                    choices=['rf', 'gbm', 'svm', 'nn', 'xgb', 'new_model', 'all'],
                    help='Machine learning model to use')
```

## Performance Considerations

- **Memory Usage**: For large datasets, consider using chunking or iterative processing
- **Computation Time**: Feature extraction and hyperparameter tuning are the most computationally intensive steps
- **Parallelization**: The scikit-learn models support parallel processing with the `n_jobs` parameter

## Example Workflow

1. Prepare your FASTA file with genomic sequences
2. Run the pipeline with all models for comparison:
   ```bash
   python bioinformatics_pipeline.py --input my_sequences.fasta --output results/ --model all
   ```
3. Review the accuracy metrics to select the best model
4. Fine-tune the selected model (e.g., adjust hyperparameters)
5. Use the saved model for predictions on new data

## Common Issues and Solutions

- **Memory Error**: If encountering out of memory errors with large datasets:
  - Reduce batch size for neural networks
  - Process sequences in chunks
  - Use a subset of features

- **Slow Performance**: If the pipeline runs too slowly:
  - Reduce the number of cross-validation folds
  - Limit hyperparameter search space
  - Use a smaller k value for k-mer extraction

- **Imbalanced Data**: If your dataset has imbalanced classes:
  - Use class weights in model training
  - Consider techniques like SMOTE for oversampling

## Requirements.txt

Create a `requirements.txt` file with the following content:

```
biopython==1.81
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
xgboost>=1.5.0
tensorflow>=2.8.0
scipy>=1.7.0
```


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

