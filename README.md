MLHTHC: Multi-Layer Hypergraph Framework for Drug Interaction Prediction with Transformer and Hypergraph Convolution
A novel deep learning framework fusing Transformer and Hypergraph Convolutional Networks (HGCN) to predict drug-drug interactions (DDIs), enhancing prediction accuracy by capturing multi-dimensional drug synergies.
🌟 Project Overview

Drug-drug interactions (DDIs) are a critical issue in drug development and clinical practice. Accurate DDI prediction is vital for improving treatment safety and optimizing medication regimens. However, the exponential growth of potential drug combinations and the limitations of traditional graph-based models (which only capture binary drug relationships) hinder prediction performance.

This repository presents MLHTHC, a multi-layer hypergraph framework that integrates Transformer and hypergraph convolution to address these challenges. By modeling multi-dimensional drug synergies and fusing global-local features, MLHTHC achieves superior DDI prediction performance compared to state-of-the-art methods.

🔍 Key Challenges Addressed

- Exponential Drug Combinations: Traditional experimental verification is infeasible for the massive number of potential drug pairs.

- Limited Relationship Modeling: Conventional graph/multi-layer network models only capture binary drug interactions, failing to represent multi-dimensional synergistic effects.

- Feature Fusion Inefficiency: Isolated use of drug attributes (e.g., chemical structure, target) leads to incomplete feature representation.

⚙️ Methodology

MLHTHC consists of four core modules, forming a end-to-end DDI prediction pipeline:

1. Multi-Layer Hypergraph Construction

Build drug similarity hypergraphs based on four types of drug attributes, each forming an independent layer to capture distinct similarity patterns:

- Chemical structure similarity

- ATC code similarity

- Drug category similarity

- Target protein similarity

2. Hypergraph Importance Weighting

UseSpectral Hamming Similarity to calculate structural similarity between each constructed hypergraph and a benchmark hypergraph (built from KEGG database DDI data). The similarity scores are used as importance weights for each hypergraph layer.

3. Feature Extraction & Fusion

- Hypergraph Convolution: Employ Hypergraph Neural Network (HGNN) to extract local structural features of drug nodes from each hypergraph layer.

- Transformer Fusion: Use Transformer encoder to adaptively fuse weighted multi-layer features, capturing global long-range dependencies between drugs.

4. DDI Prediction

A Multi-Layer Perceptron (MLP) takes the fused drug pair features (concatenated features of two drugs, their absolute difference, and element-wise product) to predict the interaction probability.

📊 Experimental Results

MLHTHC was evaluated on benchmark DDI datasets, with key metrics (AUC, AUPR, F1) outperforming existing state-of-the-art methods:

- Surpasses traditional methods: DPSP, DANN, etc.

- Ablation studies confirm that the fusion of Transformer and hypergraph convolution significantly improves prediction performance.
