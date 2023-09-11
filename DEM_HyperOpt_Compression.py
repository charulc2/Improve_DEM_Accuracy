# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:11:11 2022

@author: charul
"""
from Sub_Functions.MultiLayerNet import *
from Sub_Functions.InternalEnergy import *
from Sub_Functions.IntegrationFext import *
from torch.autograd import grad
#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss

import numpy as np
import pyspark
import time
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.random as npr
import random
import os
import hyperopt as hopt


npr.seed(2019)
torch.manual_seed(2019)


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

known_right_tx = 50
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
num_test_x = 201
num_test_y = 100
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

    np.meshgrid(lin_x, lin_y)

    
    
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
    
    dom= torch.from_numpy(dom).double()
    
    return dom, boundary_neumann, boundary_dirichlet

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
                    
        #x = torch.from_numpy(data).double()
        x= data
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
       
        model_params= self.model.parameters()        
        optimizer_LBFGS = torch.optim.LBFGS(model_params, lr=learning_rate, max_iter=20,line_search_fn='strong_wolfe')

        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        loss_history= np.zeros(iteration)
        #loss_history= []
        Iter_No_Hist= []        
        
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()

                u_pred = self.getU(x,N_Layers,activatn_fn)
                
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
                                
                loss_history[t]= loss.item()
                energy_loss_array.append(energy_loss.data)
                Iter_No_Hist.append(t)
                self.lossArray.append(loss.data)
                
                return loss
            
            if t>0:
                # print('Iter: %d Loss: %.9e, loss diff=  %.4e'
                #       % (t,loss_history[t-1], np.abs(loss_history[t-1]-loss_history[t-2]) ))
                
                if (np.abs(loss_history[t-1]-loss_history[t-2])<10e-5):
                    #print('conv achieved')
                    break
            
            optimizer_LBFGS.step(closure)
        
        return loss_history[t-1]

    def evaluate_model(self, x, y, E, nu, N_Layers,activatn_fn):
                
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

def hyperopt_main(x_var):
    
    lr = x_var['x_lr']
    neuron = int(x_var['neuron'])
    CNN_dev= x_var['CNN_dev']
    rff_dev = x_var['rff_dev']
    iteration = 100
    N_Layers = 5
    act_func = 'rrelu'
    
    
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

    print('lr: %.10e\t neuron: %.3d\t CNN_Sdev: %.10e\t RNN_Sdev: %.10e\t Itertions: %.3d\t Layers: %d\t Act_fn: %s\t Loss: %.10e'
      % (lr, neuron, CNN_dev,rff_dev,iteration,N_Layers, act_func,Loss))
    return Loss
    

#-------------------------------------------- H-Opt Settings--------------------------
np.random.seed(2019)
random.seed(2019)
torch.manual_seed(2019)

#-------------------------- File to write results in ---------

#path= "Specify path"
filename="01Mar_TensTPE_4var_rrelu_5lyr_200Runs_v2.txt"
if os.path.exists(filename):
  os.remove(filename)

#-------------------------- Variable HyperParameters-----------------------------
space = {
    'x_lr': hopt.hp.loguniform('x_lr', 0, 2),
    'neuron': 2*hopt.hp.quniform('neuron', 10, 60, 1),
    'CNN_dev': hopt.hp.uniform('CNN_dev', 0, 0.2),
    'rff_dev': hopt.hp.uniform('rff_dev', 0, 0.5)
}


#------------ Custom Function for early stop -------------------------------
#def customStopCondition(x, *kwargs):
#    return len(trials_name.trials)-1-trials_name.best_trial['tid']>50, kwargs

def customStopCondition(x, *kwargs):
	if trials_name.trials[0]["result"]["status"] == "new":
	    return False, kwargs
	else:
	    return len(trials_name.trials)-1-trials_name.best_trial['tid']>50, kwargs

#------------ Run H-Opt runs in parallel-------------------------------------
parallel=0

if parallel==1:
	que_len=5
	trials_name =hopt.SparkTrials(parallelism=que_len)	
else:
	trials_name= hopt.Trials()
	que_len=1


Hopt_strt_time = time.time()


best = hopt.fmin(hyperopt_main,
            space,
            algo=hopt.tpe.suggest,
            max_evals= 200,
            trials=trials_name,
            rstate = np.random.default_rng(2019),
            early_stop_fn=customStopCondition,
            max_queue_len= que_len
            )

Hopt_total_time = time.time()- Hopt_strt_time
print(best)

#------------ Write results to a file -------------------------------------

f = open(filename, "a")

f.writelines('Time %.3d\t'%(Hopt_total_time))
f.writelines('\n')
f.writelines('lr \t neuron \t CNN_dev \t rff_dev \t Loss \n')

for opt_iterNo in range(len(trials_name._ids)):
    f.writelines('%.6e\t %.3d\t %.6e\t %.6e\t %.6e \n'
       % (trials_name.idxs_vals[1]['x_lr'][opt_iterNo], trials_name.idxs_vals[1]['neuron'][opt_iterNo]*2,
trials_name.idxs_vals[1]['CNN_dev'][opt_iterNo], trials_name.idxs_vals[1]['rff_dev'][opt_iterNo], trials_name.results[opt_iterNo]['loss']))

f.writelines('total time= %.2e' %(Hopt_total_time))
f.close()

print('total time= %.2e' %(Hopt_total_time))



