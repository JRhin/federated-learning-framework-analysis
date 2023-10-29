"""In this python module we have defined our implemetation of a simple Logistic Regression using PyTorch.
"""

import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    """Our implemetation of a Logistic Regression usign PyTorch.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1):
        super(LogisticRegression, self).__init__()

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        self.linear = nn.Linear(self.input_dim, self.output_dim)


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)




if __name__ == "__main__":
    pass
