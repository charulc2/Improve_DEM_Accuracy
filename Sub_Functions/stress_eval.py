
import torch
import numpy as np
import matplotlib.pyplot as plt

class InternalEnergy_Stress:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

    def Elastic2DGauusStress (self, u, x, dxdydz, shape, density):
              
        Ux= torch.transpose(u[:, 0].unsqueeze(1).reshape(shape[0], shape[1]), 0, 1)
        Uy= torch.transpose(u[:, 1].unsqueeze(1).reshape(shape[0], shape[1]), 0, 1)
        
        X_Cord= np.transpose(x[:, 0].reshape(shape[0], shape[1]))
        Y_Cord= np.transpose(x[:, 1].reshape(shape[0], shape[1]))
        
        
        axis=-1
        
        nd = Ux.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        
        xN1= X_Cord[:(shape[1]-1)][tuple(slice2)]
        xN2= X_Cord[1:shape[1]][tuple(slice2)]
        xN3= X_Cord[0:(shape[1]-1)][tuple(slice1)]
        xN4= X_Cord[1:shape[1]][tuple(slice1)]
        
        yN1= Y_Cord[:(shape[1]-1)][tuple(slice2)]
        yN2= Y_Cord[1:shape[1]][tuple(slice2)]
        yN3= Y_Cord[0:(shape[1]-1)][tuple(slice1)]
        yN4= Y_Cord[1:shape[1]][tuple(slice1)]
        
        UxN1= Ux[:(shape[1]-1)][tuple(slice2)]
        UxN2= Ux[1:shape[1]][tuple(slice2)]
        UxN3= Ux[0:(shape[1]-1)][tuple(slice1)]
        UxN4= Ux[1:shape[1]][tuple(slice1)]
        
        UyN1= Uy[:(shape[1]-1)][tuple(slice2)]
        UyN2= Uy[1:shape[1]][tuple(slice2)]
        UyN3= Uy[0:(shape[1]-1)][tuple(slice1)]
        UyN4= Uy[1:shape[1]][tuple(slice1)]
        
        x_Centeroid= (xN1+xN2+xN3+xN4)/4
        y_Centeroid= (yN1+yN2+yN3+yN4)/4
        
        ## Differentiation of shape functions at gauss quadrature points       
        
        dN1_dsy=np.array([[-0.394337567,-0.105662433,-0.105662433,-0.394337567],[-0.394337567,-0.394337567,-0.105662433,-0.105662433]])
        dN2_dsy=np.array([[-0.105662433,-0.394337567,-0.394337567,-0.105662433],[0.394337567,0.394337567,0.105662433,0.105662433]])
        dN3_dsy=np.array([[0.394337567,0.105662433,0.105662433,0.394337567],[-0.105662433,-0.105662433,-0.394337567,-0.394337567]])
        dN4_dsy=np.array([[0.105662433,0.394337567,0.394337567,0.105662433],[0.105662433,0.105662433,0.394337567,0.394337567]])        
       
        dx=dxdydz[0]
        dy=dxdydz[1]
        dxds=dx/2
        dydt=dy/2
        
        J= np.array([[dxds,0],[0,dydt]])
        Jinv= np.linalg.inv(J)
        detJ= np.linalg.det(J)
        
        GaussPoints=4
        #density= torch.ones(UyN1.shape)
    
        #strainEnergy_GP=torch.zeros((GaussPoints,1))
        
        
        #for gp in range(GaussPoints):
			
            # dN1_dxy_Gp=np.matmul(Jinv, np.array([dN1_dsy[0][gp],dN1_dsy[1][gp]]).reshape((2,1)))
            # dN2_dxy_Gp=np.matmul(Jinv, np.array([dN2_dsy[0][gp],dN2_dsy[1][gp]]).reshape((2,1)))
            # dN3_dxy_Gp=np.matmul(Jinv, np.array([dN3_dsy[0][gp],dN3_dsy[1][gp]]).reshape((2,1)))
            # dN4_dxy_Gp=np.matmul(Jinv, np.array([dN4_dsy[0][gp],dN4_dsy[1][gp]]).reshape((2,1)))
            
            # dUxdx_GP= dN1_dxy_Gp[0][0]*UxN1+ dN2_dxy_Gp[0][0]*UxN2+ dN3_dxy_Gp[0][0]*UxN3+ dN4_dxy_Gp[0][0]*UxN4
            # dUxdy_GP= dN1_dxy_Gp[1][0]*UxN1+ dN2_dxy_Gp[1][0]*UxN2+ dN3_dxy_Gp[1][0]*UxN3+ dN4_dxy_Gp[1][0]*UxN4
            
            # dUydx_GP= dN1_dxy_Gp[0][0]*UyN1+ dN2_dxy_Gp[0][0]*UyN2+ dN3_dxy_Gp[0][0]*UyN3+ dN4_dxy_Gp[0][0]*UyN4
            # dUydy_GP= dN1_dxy_Gp[1][0]*UyN1+ dN2_dxy_Gp[1][0]*UyN2+ dN3_dxy_Gp[1][0]*UyN3+ dN4_dxy_Gp[1][0]*UyN4
                    
        
            # #Strains at all gauss quadrature points
            # e_xx_GP= dUxdx_GP
            # e_yy_GP= dUydy_GP
            # e_xy_GP= 0.5*(dUydx_GP+dUxdy_GP)
                    
            # #Stresses at all gauss quadrature points
        
            # S_xx_GP= self.E*(e_xx_GP+ self.nu*e_yy_GP)/(1-self.nu**2)        
            # S_yy_GP= self.E*(e_yy_GP+ self.nu*e_xx_GP)/(1-self.nu**2)
            # S_xy_GP= self.E*e_xy_GP/(1+self.nu)
            
        
            # strainEnergy_GP[gp][0]= torch.sum(0.5*(e_xx_GP*S_xx_GP+ e_yy_GP*S_yy_GP+ 2*e_xy_GP*S_xy_GP))*detJ
    
        #strainEnergy= sum(strainEnergy_GP)
        
        
        ## Strain energy at GP1
        dN1_dxy_Gp1=np.matmul(Jinv, np.array([dN1_dsy[0][0],dN1_dsy[1][0]]).reshape((2,1)))
        dN2_dxy_Gp1=np.matmul(Jinv, np.array([dN2_dsy[0][0],dN2_dsy[1][0]]).reshape((2,1)))
        dN3_dxy_Gp1=np.matmul(Jinv, np.array([dN3_dsy[0][0],dN3_dsy[1][0]]).reshape((2,1)))
        dN4_dxy_Gp1=np.matmul(Jinv, np.array([dN4_dsy[0][0],dN4_dsy[1][0]]).reshape((2,1)))
           
        dUxdx_GP1= dN1_dxy_Gp1[0][0]*UxN1+ dN2_dxy_Gp1[0][0]*UxN2+ dN3_dxy_Gp1[0][0]*UxN3+ dN4_dxy_Gp1[0][0]*UxN4
        dUxdy_GP1= dN1_dxy_Gp1[1][0]*UxN1+ dN2_dxy_Gp1[1][0]*UxN2+ dN3_dxy_Gp1[1][0]*UxN3+ dN4_dxy_Gp1[1][0]*UxN4
           
        dUydx_GP1= dN1_dxy_Gp1[0][0]*UyN1+ dN2_dxy_Gp1[0][0]*UyN2+ dN3_dxy_Gp1[0][0]*UyN3+ dN4_dxy_Gp1[0][0]*UyN4
        dUydy_GP1= dN1_dxy_Gp1[1][0]*UyN1+ dN2_dxy_Gp1[1][0]*UyN2+ dN3_dxy_Gp1[1][0]*UyN3+ dN4_dxy_Gp1[1][0]*UyN4
                   
        
        #Strains at all gauss quadrature points
        e_xx_GP1= dUxdx_GP1
        e_yy_GP1= dUydy_GP1
        e_xy_GP1= 0.5*(dUydx_GP1+dUxdy_GP1)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP1= (self.E*(e_xx_GP1+ self.nu*e_yy_GP1)/(1-self.nu**2))*density        
        S_yy_GP1= (self.E*(e_yy_GP1+ self.nu*e_xx_GP1)/(1-self.nu**2))*density
        S_xy_GP1= (self.E*e_xy_GP1/(1+self.nu))*density
        
        strainEnergy_GP1= torch.sum(0.5*(e_xx_GP1*S_xx_GP1+ e_yy_GP1*S_yy_GP1+ 2*e_xy_GP1*S_xy_GP1))*detJ
        
        ## Strain energy at GP2
        dN1_dxy_Gp2=np.matmul(Jinv, np.array([dN1_dsy[0][1],dN1_dsy[1][1]]).reshape((2,1)))
        dN2_dxy_Gp2=np.matmul(Jinv, np.array([dN2_dsy[0][1],dN2_dsy[1][1]]).reshape((2,1)))
        dN3_dxy_Gp2=np.matmul(Jinv, np.array([dN3_dsy[0][1],dN3_dsy[1][1]]).reshape((2,1)))
        dN4_dxy_Gp2=np.matmul(Jinv, np.array([dN4_dsy[0][1],dN4_dsy[1][1]]).reshape((2,1)))
        
        dUxdx_GP2= dN1_dxy_Gp2[0][0]*UxN1+ dN2_dxy_Gp2[0][0]*UxN2+ dN3_dxy_Gp2[0][0]*UxN3+ dN4_dxy_Gp2[0][0]*UxN4
        dUxdy_GP2= dN1_dxy_Gp2[1][0]*UxN1+ dN2_dxy_Gp2[1][0]*UxN2+ dN3_dxy_Gp2[1][0]*UxN3+ dN4_dxy_Gp2[1][0]*UxN4
        
        dUydx_GP2= dN1_dxy_Gp2[0][0]*UyN1+ dN2_dxy_Gp2[0][0]*UyN2+ dN3_dxy_Gp2[0][0]*UyN3+ dN4_dxy_Gp2[0][0]*UyN4
        dUydy_GP2= dN1_dxy_Gp2[1][0]*UyN1+ dN2_dxy_Gp2[1][0]*UyN2+ dN3_dxy_Gp2[1][0]*UyN3+ dN4_dxy_Gp2[1][0]*UyN4
                
        
        #Strains at all gauss quadrature points
        e_xx_GP2= dUxdx_GP2
        e_yy_GP2= dUydy_GP2
        e_xy_GP2= 0.5*(dUydx_GP2+dUxdy_GP2)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP2= (self.E*(e_xx_GP2+ self.nu*e_yy_GP2)/(1-self.nu**2))*density        
        S_yy_GP2= (self.E*(e_yy_GP2+ self.nu*e_xx_GP2)/(1-self.nu**2))*density
        S_xy_GP2= (self.E*e_xy_GP2/(1+self.nu))*density
        
        strainEnergy_GP2= torch.sum(0.5*(e_xx_GP2*S_xx_GP2+ e_yy_GP2*S_yy_GP2+ 2*e_xy_GP2*S_xy_GP2))*detJ
        
        ## Strain energy at GP3
        dN1_dxy_GP3=np.matmul(Jinv, np.array([dN1_dsy[0][2],dN1_dsy[1][2]]).reshape((2,1)))
        dN2_dxy_GP3=np.matmul(Jinv, np.array([dN2_dsy[0][2],dN2_dsy[1][2]]).reshape((2,1)))
        dN3_dxy_GP3=np.matmul(Jinv, np.array([dN3_dsy[0][2],dN3_dsy[1][2]]).reshape((2,1)))
        dN4_dxy_GP3=np.matmul(Jinv, np.array([dN4_dsy[0][2],dN4_dsy[1][2]]).reshape((2,1)))
        
        dUxdx_GP3= dN1_dxy_GP3[0][0]*UxN1+ dN2_dxy_GP3[0][0]*UxN2+ dN3_dxy_GP3[0][0]*UxN3+ dN4_dxy_GP3[0][0]*UxN4
        dUxdy_GP3= dN1_dxy_GP3[1][0]*UxN1+ dN2_dxy_GP3[1][0]*UxN2+ dN3_dxy_GP3[1][0]*UxN3+ dN4_dxy_GP3[1][0]*UxN4
        
        dUydx_GP3= dN1_dxy_GP3[0][0]*UyN1+ dN2_dxy_GP3[0][0]*UyN2+ dN3_dxy_GP3[0][0]*UyN3+ dN4_dxy_GP3[0][0]*UyN4
        dUydy_GP3= dN1_dxy_GP3[1][0]*UyN1+ dN2_dxy_GP3[1][0]*UyN2+ dN3_dxy_GP3[1][0]*UyN3+ dN4_dxy_GP3[1][0]*UyN4
                
        
        #Strains at all gauss quadrature points
        e_xx_GP3= dUxdx_GP3
        e_yy_GP3= dUydy_GP3
        e_xy_GP3= 0.5*(dUydx_GP3+dUxdy_GP3)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP3= (self.E*(e_xx_GP3+ self.nu*e_yy_GP3)/(1-self.nu**2))*density        
        S_yy_GP3= (self.E*(e_yy_GP3+ self.nu*e_xx_GP3)/(1-self.nu**2))*density
        S_xy_GP3= (self.E*e_xy_GP3/(1+self.nu))*density
        
        strainEnergy_GP3= torch.sum(0.5*(e_xx_GP3*S_xx_GP3+ e_yy_GP3*S_yy_GP3+ 2*e_xy_GP3*S_xy_GP3))*detJ
        
        ## Strain energy at GP4
        dN1_dxy_GP4=np.matmul(Jinv, np.array([dN1_dsy[0][3],dN1_dsy[1][3]]).reshape((2,1)))
        dN2_dxy_GP4=np.matmul(Jinv, np.array([dN2_dsy[0][3],dN2_dsy[1][3]]).reshape((2,1)))
        dN3_dxy_GP4=np.matmul(Jinv, np.array([dN3_dsy[0][3],dN3_dsy[1][3]]).reshape((2,1)))
        dN4_dxy_GP4=np.matmul(Jinv, np.array([dN4_dsy[0][3],dN4_dsy[1][3]]).reshape((2,1)))
        
        dUxdx_GP4= dN1_dxy_GP4[0][0]*UxN1+ dN2_dxy_GP4[0][0]*UxN2+ dN3_dxy_GP4[0][0]*UxN3+ dN4_dxy_GP4[0][0]*UxN4
        dUxdy_GP4= dN1_dxy_GP4[1][0]*UxN1+ dN2_dxy_GP4[1][0]*UxN2+ dN3_dxy_GP4[1][0]*UxN3+ dN4_dxy_GP4[1][0]*UxN4
        
        dUydx_GP4= dN1_dxy_GP4[0][0]*UyN1+ dN2_dxy_GP4[0][0]*UyN2+ dN3_dxy_GP4[0][0]*UyN3+ dN4_dxy_GP4[0][0]*UyN4
        dUydy_GP4= dN1_dxy_GP4[1][0]*UyN1+ dN2_dxy_GP4[1][0]*UyN2+ dN3_dxy_GP4[1][0]*UyN3+ dN4_dxy_GP4[1][0]*UyN4
                
        
        #Strains at all gauss quadrature points
        e_xx_GP4= dUxdx_GP4
        e_yy_GP4= dUydy_GP4
        e_xy_GP4= 0.5*(dUydx_GP4+dUxdy_GP4)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP4= (self.E*(e_xx_GP4+ self.nu*e_yy_GP4)/(1-self.nu**2))*density        
        S_yy_GP4= (self.E*(e_yy_GP4+ self.nu*e_xx_GP4)/(1-self.nu**2))*density
        S_xy_GP4= (self.E*e_xy_GP4/(1+self.nu))*density
        
        S_xx= ((S_xx_GP1+ S_xx_GP2+ S_xx_GP3+ S_xx_GP4)/4).detach().numpy()
        S_yy= (S_yy_GP1+ S_yy_GP2+ S_yy_GP3+ S_yy_GP4).detach().numpy()/4
        S_xy= (S_xy_GP1+ S_xy_GP2+ S_xy_GP3+ S_xy_GP4).detach().numpy()/4
        
        S_v=np.sqrt(1/2*(np.square(S_xx-S_yy)+np.square(S_xx)+np.square(S_yy)+6*np.square(S_xy)))
        S_v_Gp1=torch.sqrt(1/2*(torch.square(S_xx_GP1-S_yy_GP1)+torch.square(S_xx_GP1)+torch.square(S_yy_GP1)+6*torch.square(S_xy_GP1))).detach().numpy()
        S_v_Gp2=torch.sqrt(1/2*(torch.square(S_xx_GP2-S_yy_GP2)+torch.square(S_xx_GP2)+torch.square(S_yy_GP2)+6*torch.square(S_xy_GP2))).detach().numpy()
        S_v_Gp3=torch.sqrt(1/2*(torch.square(S_xx_GP3-S_yy_GP3)+torch.square(S_xx_GP3)+torch.square(S_yy_GP3)+6*torch.square(S_xy_GP3))).detach().numpy()
        S_v_Gp4=torch.sqrt(1/2*(torch.square(S_xx_GP4-S_yy_GP4)+torch.square(S_xx_GP4)+torch.square(S_yy_GP4)+6*torch.square(S_xy_GP4))).detach().numpy()
        
        vmin= 0
        vmax= 40
        
        fig,ax10=plt.subplots(1,1)
        cp= ax10.tricontourf(x_Centeroid.reshape(x_Centeroid.size,),y_Centeroid.reshape(x_Centeroid.size,),
                             S_v.reshape(x_Centeroid.size,),
                             12, levels=np.linspace(vmin, vmax,12),cmap='jet')
        ax10.set_title('DEM: $S_{Mises}$')   
        ax10.set_aspect(4.5/4)
        fig.colorbar(cp,shrink=0.8,location = 'left')  
        plt.show()
        
        fig,ax10=plt.subplots(1,1)
        cp= ax10.tricontourf(x_Centeroid.reshape(x_Centeroid.size,),y_Centeroid.reshape(x_Centeroid.size,),
                             S_xx_GP3.detach().numpy().reshape(x_Centeroid.size,),12,cmap='jet')
        ax10.set_title('DEM: $S_{11}$')   
        ax10.set_aspect(4.5/4)
        fig.colorbar(cp,shrink=0.8,location = 'left')  
        plt.show()
        
        fig,ax10=plt.subplots(1,1)
        cp= ax10.tricontourf(x_Centeroid.reshape(x_Centeroid.size,),y_Centeroid.reshape(x_Centeroid.size,),
                             S_yy_GP2.detach().numpy().reshape(x_Centeroid.size,),12,cmap='jet')
        ax10.set_title('DEM: $S_{22}$')   
        ax10.set_aspect(4.5/4)
        fig.colorbar(cp,shrink=0.8,location = 'left')  
        plt.show()
        
        
        fig,ax10=plt.subplots(1,1)
        cp= ax10.tricontourf(X_Cord.detach().numpy().reshape(X_Cord.shape[0]*X_Cord.shape[1],),
                             Y_Cord.detach().numpy().reshape(X_Cord.shape[0]*X_Cord.shape[1],),
                             Ux.detach().numpy().reshape(X_Cord.shape[0]*X_Cord.shape[1],)
                             ,12,cmap='jet')
        ax10.set_title('DEM: $U_{x}$')   
        ax10.set_aspect(4.5/4)
        fig.colorbar(cp,shrink=0.8,location = 'left')  
        plt.show()
        
        
        
        strainEnergy_GP4= torch.sum(0.5*(e_xx_GP4*S_xx_GP4+ e_yy_GP4*S_yy_GP4+ 2*e_xy_GP4*S_xy_GP4))*detJ
        

        
        strainEnergy= strainEnergy_GP1 +strainEnergy_GP2 +strainEnergy_GP3 +strainEnergy_GP4
        
        return strainEnergy
