import torch
import numpy as np

class IntegrationFext:
    def __init__(self, dim):
        self.dim = dim

    def lossFextEnergy(self, u,x, neuBC_coordinates, neuBC_values, neuBC_idx, dxdydz):
        a=1
        
        dx=dxdydz[0]
        dy=dxdydz[1]
        dxds=dx/2
        dydt=dy/2
        
        J= np.array([[dxds,0],[0,dydt]])
        Jinv= np.linalg.inv(J)
        detJ= np.linalg.det(J)
        
        #neuPt_u= u[neuBC_idx[0].numpy()]
        neuPt_u=u[neuBC_idx[0][0].long()]
        fext=neuPt_u*neuBC_values[0]*dy
        fext[-1]=fext[-1]/2
        fext[0]=fext[0]/2
        
        FextEnergy=torch.sum(fext)
        
        
        return FextEnergy
        # Change return function here 
        #return self.approxIntegration(f, x, dx, dy, dz, shape)

    # def approxIntegration(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None):
    #     y = f.reshape(shape[0], shape[1])
    #     axis=-1
        
    #     nd = y.ndimension()
    #     slice1 = [slice(None)] * nd
    #     slice2 = [slice(None)] * nd
    #     slice1[axis] = slice(1, None)
    #     slice2[axis] = slice(None, -1)
        
    #     a1= y[:(shape[0]-1)][tuple(slice2)]
    #     a2= y[1:shape[0]][tuple(slice2)]
    #     a3= y[0:(shape[0]-1)][tuple(slice1)]
    #     a4= y[1:shape[0]][tuple(slice1)]
        
    #     b1= (a1+a2+a3)/6*dx*dy
    #     b2= (a3+a4+a2)/6*dx*dy
        
    #     b=b1+b2
    #     c= torch.sum(b)
        
    #     return c