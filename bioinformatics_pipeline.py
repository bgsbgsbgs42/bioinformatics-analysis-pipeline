#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bioinformatics Analysis Pipeline

This script implements an advanced bioinformatics analysis pipeline for genomic sequence analysis.
It includes data preprocessing, feature extraction, feature normalization, model training,
evaluation, and visualization.

Usage:
    python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/

"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Bioinformatics Analysis Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Path to input FASTA file')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (proportion)')
    parser.add_argument('--k', type=int, default=3, help='K-mer size for feature extraction')
    parser.add_argument('--model', type=str, default='rf', 
                        choices=['rf', 'gbm', 'svm', 'nn', 'xgb', 'all'],
                        help='Machine learning model to use')
    return parser.parse_args()


def load_sequences(file_path):
    """Load sequences from a FASTA file.
    
    Args:
        file_path (str): Path to the FASTA file
        
    Returns:
        list: List of sequence strings
    """
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences


def get_kmers(sequence, k=3):
    """Extract k-mers from a sequence.
    
    Args:
        sequence (str): Input sequence
        k (int): Size of k-mer
        
    Returns:
        list: List of k-mers
    """
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def kmer_frequency(sequences, k=3):
    """Count k-mer frequencies across all sequences.
    
    Args:
        sequences (list): List of sequence strings
        k (int): Size of k-mer
        
    Returns:
        dict: Dictionary of k-mer counts
    """
    kmer_counts = {}
    for seq in sequences:
        kmers = get_kmers(seq, k)
        for kmer in kmers:
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
            else:
                kmer_counts[kmer] = 1
    return kmer_counts


def create_feature_matrix(sequences, kmer_counts, k=3):
    """Create a feature matrix from k-mer counts.
    
    Args:
        sequences (list): List of sequence strings
        kmer_counts (dict): Dictionary of k-mer counts
        k (int): Size of k-mer
        
    Returns:
        numpy.ndarray: Feature matrix
    """
    features = []
    kmer_list = list(kmer_counts.keys())
    
    for seq in sequences:
        kmers = get_kmers(seq, k)
        feature_vector = [kmers.count(kmer) for kmer in kmer_list]
        features.append(feature_vector)
    
    return np.array(features)


def amino_acid_composition(sequence):
    """Calculate amino acid composition of a sequence.
    
    Args:
        sequence (str): Input sequence
        
    Returns:
        dict: Dictionary of amino acid frequencies
    """
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    composition = {aa: sequence.count(aa) / len(sequence) if len(sequence) > 0 else 0 
                   for aa in aa_list}
    return composition


def calculate_hydrophobicity(sequence):
    """Calculate average hydrophobicity of a sequence.
    
    Args:
        sequence (str): Input sequence
        
    Returns:
        float: Average hydrophobicity score
    """
    # Hydrophobicity index
    hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                      'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                      'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                      'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
    
    if len(sequence) == 0:
        return 0
    
    return sum(hydrophobicity.get(aa, 0) for aa in sequence) / len(sequence)


def extract_features(sequences, k=3):
    """Extract multiple feature sets from sequences.
    
    Args:
        sequences (list): List of sequence strings
        k (int): Size of k-mer
        
    Returns:
        pandas.DataFrame: Combined feature dataframe
    """
    # K-mer features
    kmer_counts = kmer_frequency(sequences, k)
    kmer_features = create_feature_matrix(sequences, kmer_counts, k)
    
    # Create a DataFrame with k-mer features
    kmer_feature_names = list(kmer_counts.keys())
    kmer_df = pd.DataFrame(kmer_features, columns=kmer_feature_names)
    
    # Amino acid composition
    aa_compositions = [amino_acid_composition(seq) for seq in sequences]
    aa_df = pd.DataFrame(aa_compositions)
    
    # Hydrophobicity scores
    hydrophobicity_scores = [calculate_hydrophobicity(seq) for seq in sequences]
    hydro_df = pd.DataFrame({'Hydrophobicity': hydrophobicity_scores})
    
    # Combine all features
    combined_features = pd.concat([kmer_df, aa_df, hydro_df], axis=1)
    
    return combined_features


def generate_dummy_labels(n_samples):
    """Generate dummy labels for demonstration purposes.
    
    In a real scenario, these would be real labels from your dataset.
    
    Args:
        n_samples (int): Number of samples
        
    Returns:
        numpy.ndarray: Array of binary labels
    """
    return np.random.randint(2, size=n_samples)


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        RandomForestClassifier: Trained model
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Best Random Forest parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_


def train_gradient_boosting(X_train, y_train):
    """Train a Gradient Boosting Machine classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        GradientBoostingClassifier: Trained model
    """
    gbm = GradientBoostingClassifier(random_state=42)
    gbm.fit(X_train, y_train)
    return gbm


def train_svm(X_train, y_train):
    """Train a Support Vector Machine classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        SVC: Trained model
    """
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    return svm


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        XGBClassifier: Trained model
    """
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)
    return xgb


def train_neural_network(X_train, y_train, X_val=None, y_val=None):
    """Train a neural network classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        tensorflow.keras.models.Sequential: Trained model
    """
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, 
              validation_data=validation_data, verbose=1)
    
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    if model_name == 'nn':
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
    else:
        y_pred = model.predict(X_test)
    
    accuracy = np.mean(y_pred == y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n{model_name.upper()} Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }


def plot_feature_importance(model, feature_names, output_dir, model_name):
    """Plot feature importance for tree-based models.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        output_dir (str): Output directory
        model_name (str): Name of the model
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not support feature importance visualization")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features or all if less than 20
    n_features = min(20, len(feature_names))
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {n_features} Feature Importances - {model_name.upper()}")
    plt.bar(range(n_features), importances[indices[:n_features]], align="center")
    plt.xticks(range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"))
    plt.close()


def save_model(model, output_dir, model_name):
    """Save a trained model.
    
    Args:
        model: Trained model
        output_dir (str): Output directory
        model_name (str): Name of the model
    """
    if model_name == 'nn':
        model.save(os.path.join(output_dir, f"{model_name}_model"))
    else:
        joblib.dump(model, os.path.join(output_dir, f"{model_name}_model.pkl"))


def main():
    """Main function to run the bioinformatics analysis pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load sequences
    print(f"Loading sequences from {args.input}...")
    sequences = load_sequences(args.input)
    print(f"Loaded {len(sequences)} sequences")
    
    # Extract features
    print(f"Extracting features with k={args.k}...")
    features_df = extract_features(sequences, k=args.k)
    print(f"Extracted {features_df.shape[1]} features")
    
    # Get feature names
    feature_names = features_df.columns.tolist()
    
    # Generate dummy labels (replace with actual labels in a real scenario)
    print("Generating labels...")
    labels = generate_dummy_labels(len(sequences))
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_df)
    
    # Save the scaler for future use
    joblib.dump(scaler, os.path.join(args.output, 'feature_scaler.pkl'))
    
    # Split data
    print(f"Splitting data with test_size={args.test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels, test_size=args.test_size, random_state=42
    )
    
    # Further split training data for neural network validation
    X_train_nn, X_val, y_train_nn, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train and evaluate models
    models_to_train = []
    if args.model == 'all':
        models_to_train = ['rf', 'gbm', 'svm', 'xgb', 'nn']
    else:
        models_to_train = [args.model]
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\nTraining {model_name.upper()} model...")
        
        if model_name == 'rf':
            model = train_random_forest(X_train, y_train)
        elif model_name == 'gbm':
            model = train_gradient_boosting(X_train, y_train)
        elif model_name == 'svm':
            model = train_svm(X_train, y_train)
        elif model_name == 'xgb':
            model = train_xgboost(X_train, y_train)
        elif model_name == 'nn':
            model = train_neural_network(X_train_nn, y_train_nn, X_val, y_val)
        
        # Evaluate model
        eval_results = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = eval_results
        
        # Plot feature importance for tree-based models
        if model_name in ['rf', 'gbm', 'xgb']:
            plot_feature_importance(model, feature_names, args.output, model_name)
        
        # Save model
        save_model(model, args.output, model_name)
    
    # Compare models if multiple models were trained
    if len(models_to_train) > 1:
        print("\nModel Comparison:")
        for model_name, result in results.items():
            print(f"{model_name.upper()}: Accuracy = {result['accuracy']:.4f}")
    
    print(f"\nAnalysis complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
