import sys
import os
import random

import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
import torch.nn as nn

from .utils import toNP
from .geometry import to_basis, from_basis


# class PR(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1):
#         super(PR, self).__init__()

#         self.modes1 = modes1
#         self.scale = (1 / (in_channels*out_channels))
#         self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
#         self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
       
#     def output_PR(self, lambda1,alpha, weights_pole, weights_residue):   
#         Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
#         term1=torch.div(1,torch.sub(lambda1,weights_pole))
#         Hw=weights_residue*term1
#         output_residue1=torch.einsum("bix,xiok->box", alpha, Hw) 
#         output_residue2=torch.einsum("bix,xiok->bok", alpha, -Hw) 
#         return output_residue1,output_residue2    

#     def forward(self, x):
#         t=grid_x_train.cuda()
#         #Compute input poles and resudes by FFT
#         dt=(t[1]-t[0]).item()
#         alpha = torch.fft.fft(x)
#         lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
#         lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         lambda1=lambda1.cuda()
    
#         # Obtain output poles and residues for transient part and steady-state part
#         output_residue1,output_residue2= self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)
    
#         # Obtain time histories of transient response and steady-state response
#         x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
#         x1 = torch.real(x1)
#         x2=torch.zeros(output_residue2.shape[0],output_residue2.shape[1],t.shape[0], device=alpha.device, dtype=torch.cfloat)    
#         term1=torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
#         term2=torch.exp(term1) 
#         x2=torch.einsum("bix,ioxz->boz", output_residue2,term2)
#         x2=torch.real(x2)
#         x2=x2/x.size(-1)
#         return x1+x2
    

class LearnedTimeDiffusion_LNO(nn.Module):
    """
    Applies diffusion with learned per-channel t.

    In the spectral domain this becomes 
        f_out = e ^ (lambda_i t) f_in

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values 
    """

    def __init__(self, C_inout, method='spectral', num_poles = 4):
        super(LearnedTimeDiffusion_LNO, self).__init__()
        self.C_inout = C_inout
        self.in_channels = C_inout
        self.out_channels = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method # one of ['spectral', 'implicit_dense']

        # 系统极点和留数作为可学习参数 (对应原始PR类中的weights_pole和weights_residue)
        #self.scale = (1 / (C_inout))
        # 系统极点和留数 (对应原始代码中的weights_pole和weights_residue)
        self.system_poles = nn.Parameter(torch.randn(C_inout, 8, num_poles, dtype=torch.cfloat))
        self.system_residues = nn.Parameter(torch.randn(C_inout, 8, num_poles, dtype=torch.cfloat))

        nn.init.constant_(self.diffusion_time, 0.0)

    def output_PR(self, evals, x_spec, system_poles, system_residues):
        """
        极点-留数运算 (对应原始PR类中的output_PR方法)
        
        参数:
            evals: LBO特征值，形状为 (num_eigenvectors,)
            x_spec: 输入谱系数，形状为 (batch_size, in_channels, num_eigenvectors)
            system_poles: 系统极点，形状为 (in_channels, out_channels, num_poles)
            system_residues: 系统留数，形状为 (in_channels, out_channels, num_poles)
            
        返回:
            output_residue1: 稳态响应留数
            output_residue2: 瞬态响应留数
        """
        # batch_size, in_channels, num_eigenvectors = x_spec.shape
        # out_channels = system_poles.shape[1]
        
        # 扩展维度以便广播计算
        # evals_extended: (1, 1, 1, num_eigenvectors)
        evals_extended = torch.sqrt(torch.abs(evals)).unsqueeze(0).unsqueeze(0)
        #print(evals.shape)
        
        # system_poles_extended: (in_channels, out_channels, num_poles, 1)
        system_poles_extended = system_poles.unsqueeze(-1)
        
        # 计算Hw = residue / (eval - pole) (对应原始PR类中的Hw计算)
        # Hw = system_residues.unsqueeze(-1) / (evals_extended - system_poles_extended)
        term1 = torch.div(1, torch.sub(evals_extended, system_poles_extended))
        Hw = system_residues.unsqueeze(-1) * term1
        
        # 计算output_residue1 = α * H(λ) (对应稳态响应)
        # 使用einsum代替原始代码中的矩阵乘法
        #print("x_spec:", x_spec.shape, "Hw:", Hw.shape)
        # output_residue1 = α * H(λ) (稳态响应)
        output_residue1 = torch.einsum("bix,iokx->box", x_spec, Hw) 
        # output_residue2 = α * H(μ) (瞬态响应)
        output_residue2 = torch.einsum("bix,iokx->bik", x_spec, -Hw) 
        
        return output_residue1, output_residue2
    
    def compute_basis(self, tx):  # [T]
        """Compute basis functions φ_i(t) = e^{-σ_i t} e^{iω_i t}"""
        # Compute base functions ϕ_i(t) = e^{-σ_i t}
        term1 = torch.einsum("cs,t->cst", self.sigma_i, tx)  # sigma_i=[inc,num_sigma_i],tx=[x_num,1]
        basis = torch.exp(term1)
        return basis
        

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':

            # Transform to spectral
            batch_size, num_vertices, _ = x.shape
            x_spec = to_basis(x, evecs, mass)

            ### Diffuse ###
            time = self.diffusion_time
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec
            x_diffuse = from_basis(x_diffuse_spec, evecs)

            ### LNO ###
            x_spec_lno = x_spec.transpose(1, 2).to(torch.cfloat)
            system_poles = self.system_poles
            system_residues = self.system_residues      
            output_residue1, output_residue2 = self.output_PR(
                evals, x_spec_lno, system_poles, system_residues
            )
            #print(output_residue1.shape)
            # print(output_residue2.shape)
            # Transform back to per-vertex 
            x1 = torch.real(output_residue1).transpose(1, 2)
            x_lno = from_basis(x1, evecs)
            #x1 = torch.real(x1)
            # print(x1.shape)

            # ### Diffuse ###
            # time = self.diffusion_time
            # diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            # x_diffuse_spec = diffusion_coefs * x_spec
            # Transform back to per-vertex 
            # x_diffuse = from_basis(x_diffuse_spec, evecs)

            tx = torch.arange(1, num_vertices + 1, dtype=torch.float32, device=x.device).unsqueeze(0) / num_vertices
            term1 = torch.einsum("biof,kx->biox", self.system_poles.unsqueeze(0), tx.type(torch.complex64).reshape(1, -1))
            term2 = torch.exp(term1)
            # print(term2.shape)
            # print(output_residue2.shape)
            x2 = torch.einsum("bix,bioz->boz", output_residue2, term2)
            x2 = torch.real(x2) / num_vertices
            x_lno = x_lno + x2.transpose(1, 2)
            #print(x_diffuse.shape)
            
        elif self.method == 'implicit_dense':
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse, x_lno
    


class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.

    In the spectral domain this becomes 
        f_out = e ^ (lambda_i t) f_in

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values 
    """

    def __init__(self, C_inout, method='spectral'):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method # one of ['spectral', 'implicit_dense']

        nn.init.constant_(self.diffusion_time, 0.0)
        

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        if self.method == 'spectral':

            # Transform to spectral
            print(x.shape)
            print(evecs.shape)
            print(mass.shape)
            x_spec = to_basis(x, evecs, mass)
            print(x_spec.shape)

            # Diffuse
            time = self.diffusion_time
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            print(diffusion_coefs.shape)
            x_diffuse_spec = diffusion_coefs * x_spec
            print(x_diffuse_spec.shape)

            # Transform back to per-vertex 
            x_diffuse = from_basis(x_diffuse_spec, evecs)
            print(x_diffuse.shape)
            
        elif self.method == 'implicit_dense':
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)
            
            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")


        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.
    
    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots 
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if(self.with_gradient_rotations):
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[...,0]) - self.A_im(vectors[...,1])
            vectorsBimag = self.A_re(vectors[...,1]) + self.A_im(vectors[...,0])
        else:
            vectorsBreal = self.A(vectors[...,0])
            vectorsBimag = self.A(vectors[...,1])

        dots = vectorsA[...,0] * vectorsBreal + vectorsA[...,1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True, 
                 diffusion_method='spectral',
                 with_gradient_features=True, 
                 with_gradient_rotations=True):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion_LNO(self.C_width, method=diffusion_method)
        
        self.MLP_C = 2*self.C_width+8
      
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width, with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width
        
        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)


    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0] # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        
        # Diffusion block 
        x_diffuse, x_lno = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
                x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad) 

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_lno, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_lno, x_diffuse), dim=-1)

        
        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out


class DiffusionNet(nn.Module):

    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, outputs_at='vertices', mlp_hidden_dims=None, dropout=True, 
                       with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral'):   
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(C_width = C_width,
                                      mlp_hidden_dims = mlp_hidden_dims,
                                      dropout = dropout,
                                      diffusion_method = diffusion_method,
                                      with_gradient_features = with_gradient_features, 
                                      with_gradient_rotations = with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)
        
        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == 'vertices': 
            x_out = x
        
        elif self.outputs_at == 'edges': 
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        
        elif self.outputs_at == 'faces': 
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)
        
        elif self.outputs_at == 'global_mean': 
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant. 
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out
