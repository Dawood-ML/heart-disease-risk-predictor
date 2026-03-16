import torch
from torch import nn
from typing import List

# WHAT: A configurable Multi-Layer Perceptron for tabular binary classification
# WHY: MLPs are the right tool for tabular data. CNNs are for images, LSTMs for
#      sequences. Tabular data = dense layers. Simple, fast, interpretable.
# WHEN: Tabular classification with <10k rows and <50 features — MLP is your first try.
# WHEN NOT: When you have millions of rows and complex feature interactions — then
#           tree ensembles (XGBoost) often outperform. Neural nets shine when you have
#           more data, need embeddings, or need to share representations across tasks.


class HeartDiseaseClassifier(nn.Module):
    """
    Configurable MLP for heart disease binary classification.

    Architecture: Input → [Linear → BatchNorm → ReLU → Dropout] x N → Output

    Why BatchNorm?
        Normalizes activations between layers. Stabilizes training,
        allows higher learning rates, reduces sensitivity to initialization.

    Why Dropout?
        Randomly zeros out neurons during training. Forces the network to
        learn redundant representations. Reduces overfitting on small datasets
        like this one (303 samples).

    Why this layer structure?
        Decreasing hidden dims [64, 32, 16] creates an information bottleneck.
        The network learns to compress 13 clinical features into progressively
        more abstract representations, then outputs a single risk score.
    """

    def __init__(
            self, 
            input_dim: int,
            hidden_dims: List[int],
            dropout_rate: float=0.3,
            activation: str = 'relu'
    ):
        super().__init__()
        # WHAT: Choose activation function from config
        # WHY: Keep architecture configurable — we log this as a param in MLflow
        #      so we can compare relu vs leaky_relu experiments in the UI
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU()
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(activations.keys())}")
        
        # Build layers dynamically from hidden_dims config
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activations[activation],
                nn.Dropout(p=dropout_rate)
            ])

        # Output layer - single neuron, no activation
        # output is raw logit pre sigmoid
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

        # Initialize xavier weights
        
