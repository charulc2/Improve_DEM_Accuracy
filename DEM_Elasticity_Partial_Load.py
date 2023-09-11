# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:11:11 2022

@author: charul
"""

#import matplotlib as plt

# from Sub_Functions import define_structure as des
from Sub_Functions.MultiLayerNet import *
from Sub_Functions.InternalEnergy import *
from Sub_Functions.IntegrationFext import *
from torch.autograd import grad

import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy.random as npr
import random

npr.seed(2019)
torch.manual_seed(2019)

#-------------------------- Variable HyperParameters-----------------------------
# lr= 1.351451
# neuron=106
# CNN_dev= 0.01977339
# rff_dev = 0.4609444
# iteration = 100
# N_Layers = 4
# act_func = 'rrelu'


######----- Optimal HyperParameters: Compression------#######
lr= 1.351451
neuron=106
CNN_dev= 0.0197734
rff_dev = 0.4609444
iteration = 100
N_Layers = 5
act_func = 'rrelu'

######----- Optimal HyperParameters: Tenssion------#######
#lr= 1.454009
#neuron=94
#CNN_dev= 0.0150147
#rff_dev = 0.4847299
#iteration = 100
#N_Layers = 5
#act_func = 'rrelu'

######----- Optimal HyperParameters: Bending ------#######
#lr= 1.40475
#neuron=98
#CNN_dev= 0.03276142
#rff_dev = 0.4981533
#iteration = 100
#N_Layers = 5
#act_func = 'rrelu'


filename="Partial_Loadind"

#------------------------- Constant Network Parameters ----------------
D_in = 2
D_out = 2

# -------------------------- Structural Parameters ---------------------
Length = 4
Height = 1
Depth = 1.0

# -------------------------- Boundary Conditions ------------------------
known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_tx = -10
known_right_ty = 0
bc_right_penalty = 1.0

# -------------------------- Material Parameters -----------------------
model_energy = 'Elastic2D'
E = 1000
nu = 0.3


# ------------------------- Datapoints for training ---------------------
Nx = 100
Ny = 25
x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------- Datapoints for evaluation -----------------
Length_test=4
Height_test=1
num_test_x = 161
num_test_y = 41
hx_test = Length / (num_test_x - 1)
hy_test = Height / (num_test_y - 1)
shape_test=[num_test_x,num_test_y]





dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    device_string = 'cuda'
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device_string = 'cpu'
    print("CUDA not available, running on CPU")
    torch.set_default_tensor_type('torch.DoubleTensor')

mpl.rcParams['figure.dpi'] = 350



def get_Train_domain():
    x_dom = x_min, Length, Nx
    y_dom = y_min, Height, Ny
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    dom = np.zeros((Nx * Ny, 2)) #Initializing the domain array
    c = 0
    
    node_dy= (y_dom[1]-y_dom[0])/(y_dom[2]-1)
    node_dx= (x_dom[1]-x_dom[0])/(x_dom[2]-1)
    
    # Assign nodal coordinates to all the points in the dom array
    for x in np.nditer(lin_x):
        tb = y_dom[2] * c
        te = tb + y_dom[2]
        c += 1
        dom[tb:te, 0] = x
        dom[tb:te, 1] = lin_y
    #print(dom.shape)
    
    # Plot the points defined in Dom
    np.meshgrid(lin_x, lin_y)
    fig = plt.figure(figsize=(Length+1, Height+1))
    ax = fig.add_subplot(111)
    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.set_xlabel('X', fontsize=3)
    ax.set_ylabel('Y', fontsize=3)
    ax.tick_params(labelsize=4)
    
    
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx = np.where(dom[:, 0] == x_min) # Index/ node numbers at which x=x_min
    bcl_u_pts = dom[bcl_u_pts_idx, :][0] #Coordinates at which x=xmin
    bcl_u = np.ones(np.shape(bcl_u_pts)) * [known_left_ux, known_left_uy] #Define displacement constraints at the nodes
    #known_left_ux and known_left_uy are defined in config.py file

    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx = np.where(dom[:, 0] == Length)
    bcr_t_pts = dom[bcr_t_pts_idx, :][0]
    bcr_t = np.ones(np.shape(bcr_t_pts)) * [known_right_tx, known_right_ty]
    
    boundary_neumann = {
        # condition on the right
        "neumann_1": {
            "coord": bcr_t_pts,
            "known_value": bcr_t,
            "penalty": bc_right_penalty,
            "idx":np.asarray(bcr_t_pts_idx)
        }
        # adding more boundary condition here ...
    }
    
    boundary_dirichlet = {
        # condition on the left
        "dirichlet_1": {
            "coord": bcl_u_pts,
            "known_value": bcl_u,
            "penalty": bc_left_penalty,
            "idx":np.asarray(bcl_u_pts_idx)
        }
        # adding more boundary condition here ...
    }
    
    #------------------------- Partial loading ------------------------------------------
    # Right boundary condition to top 0.25 (Neumann BC)
    bcr_t_pts_idx_new = np.where((dom[:, 0] == Length) & (dom[:, 1] >0.75))
    bcr_t_pts_new = dom[bcr_t_pts_idx_new, :][0]
    bcr_t_new = np.ones(np.shape(bcr_t_pts_new)) * [known_right_tx, known_right_ty]
    
    
    ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    ax.scatter(bcl_u_pts[:, 0], bcl_u_pts[:, 1], s=0.5, facecolor='red')
    ax.scatter(bcr_t_pts[:, 0], bcr_t_pts[:, 1], s=0.5, facecolor='green')
    # ax.scatter(bcr_t_pts_new[:, 0], bcr_t_pts_new[:, 1], s=0.5, facecolor='black')
    plt.show()
    
    
    
    boundary_neumann_new = {
        # condition on the right
        "neumann_1": {
            "coord": bcr_t_pts_new,
            "known_value": bcr_t_new,
            "penalty": bc_right_penalty,
            "idx":np.asarray(bcr_t_pts_idx_new)
        }
        # adding more boundary condition here ...
    }
    
    # Return boundary_neumann_new for partial loading 
    
    return dom, boundary_neumann_new, boundary_dirichlet

def get_Test_datatest(Nx=num_test_x, Ny=num_test_y):

    x_dom_test = x_min, Length_test, Nx
    y_dom_test = y_min, Height_test, Ny
    # create points
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    xGrid, yGrid = np.meshgrid(x_space, y_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T), axis=1)
    return x_space, y_space, data_test

def get_density():
    # Passive Density elements for topology opt.
   density= torch.ones(Ny-1,Nx-1)
   train_x_coord= np.transpose(dom[:,0].reshape(Nx,Ny))
   train_y_coord= np.transpose(dom[:,1].reshape(Nx,Ny))
   
    #------------Passive elements for Circular Hole --------------------------------------------------------------
    
   Crcl_x=Length/2
   Crcl_y=Height/2
   E_major= 0.25
   E_minor=0.25
    
    
   for nodex in range(Nx):
       for nodey in range (Ny):
           if (((train_x_coord[nodey,nodex]-Crcl_x)/E_major)**2+((train_y_coord[nodey,nodex]-Crcl_y)/E_minor)**2<1):
                density[nodey,nodex]=0
   
   plt.imshow(density, extent = [x_min, Length, y_min, Height],cmap='gray_r')
   plt.colorbar(shrink=0.5,location = 'left')
   plt.show()
   
   return density
   

class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model, dim, E, nu, act_func, CNN_dev, rff_dev,N_Layers):
        # self.data = data
        self.model = MultiLayerNet(model[0], model[1], model[2],act_func, CNN_dev, rff_dev,N_Layers)
        self.model = self.model.to(dev)
        self.InternalEnergy= InternalEnergy(E, nu)
        self.FextLoss = IntegrationFext(dim)
        self.dim = dim
        self.lossArray = []

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate, N_Layers, activatn_fn, density):
                    
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)
        # get tensor inputs and outputs for boundary conditions
        # -------------------------------------------------------------------------------
        #                             Dirichlet BC
        # -------------------------------------------------------------------------------
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
        # -------------------------------------------------------------------------------
        #                           Neumann BC
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        neuBC_Zeros= {}
        neuBC_idx={}
        
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)
            neuBC_idx[i]=torch.from_numpy(neumannBC[keyi]['idx']).float().to(dev)
            
        
        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer_LBFGS = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20,line_search_fn='strong_wolfe')

        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        loss_history= np.zeros(iteration)
        Iter_No_Hist= []        
        self.last_iter= 0
        
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()

                u_pred = self.getU(x,N_Layers,activatn_fn)
                u_pred.double()
                
                # ---- Calculate internal and external energies------                 
                storedEnergy= self.InternalEnergy.Elastic2DGauusQuad(u_pred, x, dxdydz, shape, density)
                externalE=self.FextLoss.lossFextEnergy(u_pred, x, neuBC_coordinates, neuBC_values, neuBC_idx, dxdydz)
                energy_loss = storedEnergy-externalE
                
                # ---- Calculate error in dirchlet BC --------  
                bc_u_crit = torch.zeros((len(dirBC_coordinates)))
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getU(dirBC_coordinates[i],N_Layers,activatn_fn)
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred, dirBC_values[i])

                boundary_loss= torch.sum(bc_u_crit)
                
                loss = energy_loss + boundary_loss
                
                
                optimizer_LBFGS.zero_grad()
                loss.backward()
                
                # print('Iter: %d Loss: %.9e InternalE : %.9e ExternalE : %.9e'
                #       % (t + 1, loss.item(), storedEnergy.item(),externalE.item() ))
                
                loss_history[t]= loss.item()
                energy_loss_array.append(energy_loss.data)
                Iter_No_Hist.append(t)
                self.lossArray.append(loss.data)
                
                
                return loss
            
            if t>0:
                print('Iter: %d Loss: %.9e, loss diff=  %.5e'
                      % (t,loss_history[t-1], np.abs(loss_history[t-1]-loss_history[t-2]) ))
                
                if (np.abs(loss_history[t-1]-loss_history[t-2])<10e-8):
                    print('conv achieved')
                    break
            
            
            optimizer_LBFGS.step(closure)
        elapsed = time.time() - start_time
        plt.scatter(np.linspace(1,t,t), loss_history[0:t])
        plt.title('Convergence: Loss function')
        plt.show()
        print('Training time: %.4f' % elapsed)
        
        return loss_history[t]

    def evaluate_model(self, x, y, E, nu, N_Layers,shape_test, activatn_fn):
                
        Nx = len(x)
        Ny = len(y)
        xGrid, yGrid = np.meshgrid(x, y)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        xy = np.concatenate((np.array([x1D]).T, np.array([y1D]).T), axis=-1)
        xy_tensor = torch.from_numpy(xy).float()
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xy_tensor)
        u_pred_torch = self.getU(xy_tensor,N_Layers,activatn_fn)
        duxdxy = \
            grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                 create_graph=True, retain_graph=True)[0]
        duydxy = \
            grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                 create_graph=True, retain_graph=True)[0]

        E11 = duxdxy[:, 0].unsqueeze(1)
        E22 = duydxy[:, 1].unsqueeze(1)
        E12 = (duxdxy[:, 1].unsqueeze(1) + duydxy[:, 0].unsqueeze(1))/2
        E21 = (duxdxy[:, 1].unsqueeze(1) + duydxy[:, 0].unsqueeze(1))/2

        S11 = E/(1-nu**2)*(E11 + nu*E22)
        S22 = E/(1-nu**2)*(E22 + nu*E11)
        S12 = E*E12/(1+nu)
        S21 = E*E21/(1+nu)

        u_pred = u_pred_torch.detach().cpu().numpy()

        E11_pred = E11.detach().cpu().numpy()
        E12_pred = E12.detach().cpu().numpy()
        E21_pred = E21.detach().cpu().numpy()
        E22_pred = E22.detach().cpu().numpy()

        S11_pred = S11.detach().cpu().numpy()
        S12_pred = S12.detach().cpu().numpy()
        S21_pred = S21.detach().cpu().numpy()
        S22_pred = S22.detach().cpu().numpy()

        surUx = u_pred[:, 0].reshape(Ny, Nx, 1)
        surUy = u_pred[:, 1].reshape(Ny, Nx, 1)
        surUz = np.zeros([Nx, Ny, 1])

        surE11 = E11_pred.reshape(Ny, Nx, 1)
        surE12 = E12_pred.reshape(Ny, Nx, 1)
        surE13 = np.zeros([Nx, Ny, 1])
        surE21 = E21_pred.reshape(Ny, Nx, 1)
        surE22 = E22_pred.reshape(Ny, Nx, 1)
        surE23 = np.zeros([Nx, Ny, 1])
        surE33 = np.zeros([Nx, Ny, 1])

        surS11 = S11_pred.reshape(Ny, Nx, 1)
        surS12 = S12_pred.reshape(Ny, Nx, 1)
        surS13 = np.zeros([Nx, Ny, 1])
        surS21 = S21_pred.reshape(Ny, Nx, 1)
        surS22 = S22_pred.reshape(Ny, Nx, 1)
        surS23 = np.zeros([Nx, Ny, 1])
        surS33 = np.zeros([Nx, Ny, 1])

        SVonMises = np.float64(
            np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))

        return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(
            surS23), \
            np.float64(surS33), np.float64(surE11), np.float64(surE12), \
            np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(
            SVonMises)

    def getU(self, x,N_Layers,activatn_fn):
        u = self.model(x,N_Layers,activatn_fn).double()
        Ux = x[:, 0] * u[:, 0]
        Uy = x[:, 0] * u[:, 1]
        
        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy), -1)
        return u_pred
    
    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss

if __name__ == '__main__':
    
    npr.seed(2019)
    torch.manual_seed(2019)
    random.seed(2019)
        
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = get_Train_domain()
    x, y, datatest = get_Test_datatest()
    
    #--- Activate for circular inclusion-----
    #density= get_density()
    density=1
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
     
    dem = DeepEnergyMethod([D_in, neuron, D_out], 2, E, nu, act_func, CNN_dev,rff_dev,N_Layers)
    
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    Loss= dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, iteration, lr,N_Layers,act_func,density)

    # ----------------------------------------------------------------------
    #                   STEP 4: TEST MODEL
    # ----------------------------------------------------------------------
    Eval_Start_time = time.time()
    U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises = dem.evaluate_model(x, y, E, nu, N_Layers,shape_test, act_func)
    Eval_time = time.time() - Eval_Start_time
    print('Eval time: %.4f' % Eval_time)

    surUx, surUy, surUz = U  
    
    x_coords=datatest[:,0].reshape(shape_test[1]*shape_test[0],1)
    y_coords=datatest[:,1].reshape(shape_test[1]*shape_test[0],1)
    
    Ux = surUx[:,:,0].reshape(shape_test[0]*shape_test[1],1)
    Uy = surUy[:,:,0].reshape(shape_test[0]*shape_test[1],1)
    
    
    Uy= np.where(Uy ==0, 10**(-8), Uy)
    Ux= np.where(Ux ==0, 10**(-8), Ux)
    E11= np.where(E11 ==0, 10**(-8), E11)
    
    x_new_coords= x_coords+Ux
    y_new_coords= y_coords+Uy
    
    
    fig_aspect_r=Height_test/Length_test*4

    fig,(ax10,ax11)=plt.subplots(1,2)
    fig.tight_layout()
    cp = ax10.contourf(x_new_coords.reshape(shape_test[1],shape_test[0]),y_new_coords.reshape(shape_test[1],shape_test[0]),
                       Ux.reshape(shape_test[1],shape_test[0]),12,cmap='jet')
    ax10.set_title('DEM: $U_x$')  
    ax10.set_aspect(fig_aspect_r)
    fig.colorbar(cp,shrink=0.4,location = 'left',ax=ax10)  
    #plt.show()
    

    cp2 = ax11.contourf(x_new_coords.reshape(shape_test[1],shape_test[0]),y_new_coords.reshape(shape_test[1],shape_test[0]),
                        Uy.reshape(shape_test[1],shape_test[0]),12,cmap='jet')
    ax=Uy.reshape(shape_test[1],shape_test[0])
    ax11.set_title('DEM: $U_y$')
    ax11.set_aspect(fig_aspect_r)
    fig.colorbar(cp2,shrink=0.4,location = 'left',ax=ax11)
    #plt.savefig(filename+"DEM_Disp")

    print('Loss= %0.8f'%(Loss))
    print('Ux_Max= %0.5f, Ux_Min= %0.5f'%(np.amax(Ux),np.amin(Ux)))
    print('Uy_Max= %0.5f, Uy_Min= %0.5f'%(np.amax(Uy),np.amin(Uy)))
    print('Loss= %0.10f'%(Loss))
    # print('SVonMis_Max= %0.4f, SVonMis_Min= %0.4f'%(np.amax(SVonMises),np.amin(SVonMises)))



    import pandas as pd

    abaqus_res= pd.read_excel("/u/charulc2/HOpt_Feb2023/Abaqus_Displacements.xlsx", sheet_name= "Partial_Load_V1")
    shape_ab=[41,161]

    x_coords_ab=abaqus_res.X_orignal.to_numpy().reshape(shape_ab[1],shape_ab[0]).T
    y_coords_ab=abaqus_res.Y_orignal.to_numpy().reshape(shape_ab[1],shape_ab[0]).T
    

    Ux_ab = abaqus_res.U1.to_numpy().reshape(shape_ab[1],shape_ab[0]).T
    Uy_ab = abaqus_res.U2.to_numpy().reshape(shape_ab[1],shape_ab[0]).T

    Ux_ab= np.where(Ux_ab ==0, 10**(-8), Ux_ab)
    Uy_ab= np.where(Uy_ab ==0, 10**(-8), Uy_ab)

    x_new_coords_ab= abaqus_res.X_disp.to_numpy().reshape(shape_ab[1],shape_ab[0]).T
    y_new_coords_ab= abaqus_res.Y_disp.to_numpy().reshape(shape_ab[1],shape_ab[0]).T

    error_Ux= abs(Ux.reshape(shape_test[1],shape_test[0])-Ux_ab)
    error_Uy= abs(Uy.reshape(shape_test[1],shape_test[0])-Uy_ab)
    
    L2_Ux= np.power(error_Ux,2)
    L2_Uy= np.power(error_Uy,2)
    
    L2_norm= np.sqrt(sum(sum((L2_Ux+L2_Uy)))/(shape_ab[1]*shape_ab[0]))
    print('L2_norm= %0.8f'%(L2_norm))


#----------------- Final Plot-------------------

    colorbar_ratio=1
    fig,ax=plt.subplots(3,2)
    fig.tight_layout()
    plt.setp(ax, xticks=[0,1,2,3,4], yticks=[0,1])
    
    cp = ax[0][0].contourf(x_new_coords.reshape(shape_test[1],shape_test[0]),y_new_coords.reshape(shape_test[1],shape_test[0]),
                       Ux.reshape(shape_test[1],shape_test[0]),12,cmap='jet')
    ax[0][0].set_title('DEM: $U_x$')  
    ax[0][0].set_aspect(fig_aspect_r)
    fig.colorbar(cp,shrink=colorbar_ratio,location = 'left',ax=ax[0][0])  
    #plt.show()
    

    cp2 = ax[0][1].contourf(x_new_coords.reshape(shape_test[1],shape_test[0]),y_new_coords.reshape(shape_test[1],shape_test[0]),
                        Uy.reshape(shape_test[1],shape_test[0]),12,cmap='jet')
    ax[0][1].set_title('DEM: $U_y$')
    ax[0][1].set_aspect(fig_aspect_r)
    fig.colorbar(cp2,shrink=colorbar_ratio,location = 'left',ax=ax[0][1])
    
    cp = ax[1][0].contourf(x_new_coords_ab,y_new_coords_ab.reshape(shape_test[1],shape_test[0]),
                       Ux_ab.reshape(shape_test[1],shape_test[0]),12,cmap='jet')
    ax[1][0].set_title('FEM: $U_x$')  
    fig.colorbar(cp,shrink=colorbar_ratio,location = 'left', ax=ax[1][0])  
    ax[1][0].set_aspect(fig_aspect_r)

    cp2 = ax[1][1].contourf(x_new_coords_ab.reshape(shape_test[1],shape_test[0]),y_new_coords_ab.reshape(shape_test[1],shape_test[0]),
                        Uy_ab.reshape(shape_test[1],shape_test[0]),12,cmap='jet')
    ax[1][1].set_aspect(fig_aspect_r)    
    ax[1][1].set_title('FEM: $U_y$')
    fig.colorbar(cp2,shrink=colorbar_ratio,location = 'left', ax=ax[1][1])



    cp = ax[2][0].contourf(x_new_coords.reshape(shape_test[1],shape_test[0]),y_new_coords.reshape(shape_test[1],shape_test[0]),
                        error_Ux,12,cmap='jet')
    ax[2][0].set_title('$Error_x$ = |$U_{x(FEA)}$ - $U_{x(DEM)}$|')
    #plt.imshow(density, extent = [min(x_new_coords)[0], max(x_new_coords)[0], min(y_new_coords)[0], max(y_new_coords)[0]],cmap='gray',alpha=0.6)
    
    ax[2][0].set_aspect(fig_aspect_r)
 
    fig.colorbar(cp,shrink=colorbar_ratio,location = 'left', ax=ax[2][0])  

    cp2 = ax[2][1].contourf(x_new_coords.reshape(shape_test[1],shape_test[0]),y_new_coords.reshape(shape_test[1],shape_test[0]),
                        error_Uy,12,cmap='jet')
    ax[2][1].set_title('$Error_y$ = |$U_{y(FEA)}$ - $U_{y(DEM)}$|')
    ax[2][1].set_aspect(fig_aspect_r)
    fig.colorbar(cp2,shrink=colorbar_ratio,location = 'left', ax=ax[2][1])
    fig.dpi=700
    fig.set_size_inches(6, 6)
    plt.savefig(filename+"Displacements")




