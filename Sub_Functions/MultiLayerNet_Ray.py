import rff
import torch

class MultiLayerNet_Ray(torch.nn.Module):
    def __init__(self, D_in, H, D_out,act_func, CNN_dev, rff_dev, N_Layers):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet_Ray, self).__init__()
        
        #self.encoding = rff.layers.GaussianEncoding(sigma=rff_dev, input_size=D_in, encoded_size=H//2)
        
         ## Define loop to automate Layer definition  
        # self.encoding = rff.layers.GaussianEncoding(sigma=0.05, input_size=D_in, encoded_size=H//2)
        # self.encoding = rff.layers.PositionalEncoding(sigma=0.25, m=10)
        D_in=2
        H=80
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, H)
        self.linear6 = torch.nn.Linear(H, D_out)
        
    def forward(self, x,N_Layers,act_fn):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        activation_fn = getattr(torch,act_fn)       
        y_auto = []

        for ii in range(6):
            if ii==0:
                y_auto.append(self.encoding(x))
            elif ii==(N_Layers-1):
                y_auto.append(self.linear[-1](y_auto[-1]))
            else:
                y_auto.append(activation_fn(self.linear[ii](y_auto[ii-1])))
        
        return y_auto[-1]
