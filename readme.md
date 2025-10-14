Amazon Price Prediction from Unstructured Data
This project documents the development of a model to predict e-commerce product prices from raw, unstructured text. The objective was to reduce the Symmetric Mean Absolute Percentage Error (SMAPE) below 40.

Starting from a 56.35 SMAPE baseline, the final deep learning architecture achieved a 48.6 SMAPE. This significant improvement was driven by a rigorous, data-led engineering process detailed below.

Methodology & Experimental Progression
The final model was developed through a disciplined, multi-stage process of hypothesis testing and iterative improvement, guided by two core principles: starting with the simplest viable solution and letting data from experiments dictate the next step.

Baseline & Non-Linear Modeling: We first established that a simple linear model was insufficient. The single largest performance gain came from migrating to a powerful, non-linear LightGBM model combined with a log1p transform on the skewed price data, which immediately improved the SMAPE to ~51.9.

Hypothesis Invalidation (Failed Experiments): Systematic experiments proved that both abstract semantic embeddings (via Sentence-Transformers) and image features (via CLIP) were detrimental to performance. They added noise and degraded the SMAPE score. This critical, data-driven finding focused all subsequent efforts on text-based feature engineering.

Surgical Feature Engineering: Guided by error analysis, a suite of high-signal numerical features was systematically parsed from the text (e.g., feat_oz, feat_inch, feat_gb), steadily improving the SMAPE to ~50.4.

Final Deep Learning Architecture: The project culminated in a custom BertPriceRegressor architecture. This model fused the contextual understanding of a fine-tuned DistilBERT model with a dedicated regression head that integrated the most critical engineered numerical features (total_weight, weight_unit), achieving the final score of 48.6 SMAPE.

Setup and Instructions
1. Prerequisites

Python 3.9+

Conda (recommended for managing dependencies)

2. Clone the Repository

Bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
3. Download the Data

The training data and pre-computed embeddings are required to run the notebooks.

Download from this link: Project Data and Embeddings

Create a folder named input/ in the main project directory.

Place the downloaded train.csv, test.csv, and final_embeddings.pkl files inside the input/ folder.

Your final directory structure should look like this:

your-repository-name/
├── input/
│   ├── train.csv
│   ├── test.csv
│   └── final_embeddings.pkl
├── final_model.ipynb
├── feature_engineering.ipynb
└── README.md
4. Set Up the Environment

It is highly recommended to use a Conda environment to ensure all libraries work correctly.

Bash
# Create a new conda environment
conda create --name amazon-price python=3.9

# Activate the environment
conda activate amazon-price

# Install all necessary packages
pip install pandas numpy scikit-learn lightgbm torch transformers datasets jupyter
5. Run the Final Model

The notebook containing the final BertPriceRegressor code includes the complete process to train the champion model and produce a submission file.

Launch Jupyter Notebook: jupyter notebook

Open and run the cells in the final model's notebook.

Key Learnings & Constraints
Signal vs. Noise: The primary predictive signal resides in specific keywords, model numbers, and numerical jargon, not in abstract semantic meaning or visual data.

Hybrid Models: The best performance was achieved by combining a deep learning model's language understanding with explicitly engineered, domain-specific features.

Computational Limits: Due to local GPU/RAM constraints, exhaustive fine-tuning of large transformer models and large-scale data augmentation were not feasible. The process prioritized robust feature engineering and efficient modeling within the available compute budget.