import torch 
import torch.nn as nn


class ScatteringBatchNorm(nn.Module):
    ''' A general purpose batch normalization layer,
        designed particularly for scattering paths.
    
    Args:
        num_features: the number of scattering paths
        c: an optionally learnable multiplicative featurewise constant, used
            before taking the logarithm to gaussianize the histogram of scattering 
            path values.
    '''
    def __init__(self, num_features, c: float = None, eval_mode: bool = False):
        super().__init__()

        self.num_features = num_features 
        self.mu = torch.zeros(self.num_features)
        self.c = torch.nn.Parameter(torch.zeros(self.num_features)) if not c else c

        self.eval_mode = eval_mode

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )
        if input.shape[-1] != self.num_features:
            raise ValueError(
                "expected last dim to equal feature dim: {} (got {}D input)".format(self.num_features, input.shape[-1])
            )

    def forward(self, sx)
        self._check_input_dim(sx)
        
        if not self.eval_mode:
            batch_size = sx.shape[0]
            batch_mu =  (1 / batch_size) * torch.sum(sx, dim=0)
            
            self.mu = (1 / (batch_size + 1)) * self.mu.type_as(sx) + /
                (batch_size / (batch_size + 1)) * batch_mu

        sx = torch.log1p(torch.exp(self.c) * sx / (1e-3 + self.mu))
        return sx 

    def eval(self):
        self.eval_mode = True
        return self.train(False)