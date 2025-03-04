# Make the script executable
chmod +x bioinformatics_pipeline.py

# Basic usage with default parameters (Random Forest model)
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/

# Use Gradient Boosting Machine model
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --model gbm

# Use Neural Network model with a different k-mer size
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --model nn --k 4

# Try all available models
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --model all

# Use a different test set size
python bioinformatics_pipeline.py --input genomic_sequences.fasta --output results/ --test_size 0.3