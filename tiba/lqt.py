# Trajectory generation using Linear Quadratic Tracking (LQT) as described in:
#    @inproceedings{BerioGI2017,
#       author = {Berio, D and Calinon, S. and Fol Leymarie, F.},
#       title = {Generating Calligraphic Trajectories with Model Predictive Control},
#       booktitle = {Proceedings of Graphics Interface}, series = {GI 2017},
#       year = {2017},
#       address = {Edmonton, Canada},
#       month = {May},
#       publisher = {Canadian Human-Computer Communications Society},
#    }
#    @inproceedings{BerioMPCMOCO2017,
#       author = {Berio, D and Calinon, S. and Fol Leymarie, F.},
#       title = {Dynamic Graffiti Stylisation with Stochastic Optimal Control},
#       booktitle = {ACM Proceedings of the 4th International Conference on Movement and Computing},
#       series = {MOCO '17}, year = {2017}, month = {June}, address =
#       {London, UK},
#       publisher = {ACM}
#    }
#    @article{Berio2019,
#      title = {Interactive Generation of Calligraphic Trajectories from Gaussian Mixtures},
#      author = {Berio, Daniel and Fol Leymarie, Frederic and Calinon, Sylvain},
#      publisher = {Springer}, journal = {Mixture Models and Applications},
#      series = {Unsupervised and Semi-Supervised Learning},
#      editor="Bouguila, N. and Fan, W.",
#      pages = {23-38},
#      year = {2019},
#    }
#    Please cite one of these papers if using this code for accademic purposes.

from importlib import reload
import numpy as np
from scipy.linalg import block_diag 
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, expm, eigh
from numpy.linalg import inv, pinv, det, norm, solve
import pdb

def radians( x ):
    return np.pi/180*x

def degrees( x ):
    return x * (180.0/np.pi)

def discretize_system(A, B, dt, method='exact'):
    ''' Discretize a continous LTI system, 
        considering only dynamics matrix A and control matrix B
    ''' 
    # Make sure B is a column    
    if len(B.shape) == 1:
        B = B.reshape(-1, 1)
        
    if method=='zoh':
        # Linear Systems: A State Variable Approach with Numerical Implementation pg 215
        # adapted from scipy impl        
        EM  = np.hstack((A, B))
        zeros =  np.hstack((np.zeros((B.shape[1], B.shape[0])),
                              np.zeros((B.shape[1], B.shape[1]))))
        EM = np.vstack([EM, zeros])
        M = expm(EM*dt)
            
        Ad = M[:A.shape[0], 0:A.shape[1]]
        Bd = M[:B.shape[0], A.shape[1]:]
        return Ad, Bd
    elif method=='bilinear':
        # This parameter is used repeatedly - compute once here
        alpha = 0.5
        ima = np.eye(A.shape[0]) - alpha*dt*A
        ad = np.linalg.solve(ima, np.eye(A.shape[0]) + (1.0-alpha)*dt*A)
        bd = np.linalg.solve(ima, dt*B)
        return ad, bd
    elif method=='exact':
        # Assume canonical chain of integrators, equivalent to ZOH
        dim = A.shape[0]
        Ad = np.zeros((dim,dim))
        for i in range(dim):
            Ad = Ad + np.diag(np.ones(dim-i), i) * dt**i * 1/np.math.factorial(i)
        Bd = np.zeros((dim, 1))
        for i in range(1, dim+1):
            Bd[dim-i] = dt**i / np.math.factorial(i)
        return Ad, Bd
    else: # Euler
        A = np.eye(A.shape[0]) + A * dt
        B = B * dt
        return A, B
   

def largest_eigs(X, k):
    ''' Return k eigencomponents with decreasing eigenvalues'''
    N = X.shape[0]
    D, V = np.linalg.eig(X) #eigh(X) #, eigvals=(N-k,N-1))
    if k < 0:
        return D, V
    k = min(k, D.shape[0])
    return D[:k], V[:,:k]

def make_sigma(theta, scale):
    ''' Builds a 2d covariance with a rotation and scale'''
    scale=np.array(scale)
    
    # rotation matrix
    Phi = np.eye(2, 2)
    ct = np.cos(theta)
    st = np.sin(theta)
    Phi[0,0], Phi[0,1] = ct, -st
    Phi[1,0], Phi[1,1] = st, ct
    
    # scale matrix
    S = np.diag(scale*scale) 
    return Phi @ S @ Phi.T


def make_lambda(theta, scale):
    return np.linalg.inv(make_sigma(theta, scale))


def SHM_r(d, order, segment_duration): #dt, n, m):
    ''' simple harmonic motion based scale factor,
        computes the control cost r in terms of maximum allowed displacement,
        assuming an oscillatory motion along a segment.
        transfer function is 1/s^k, order k, evaluated for j-omega and taking complex absolute value
    '''
    omega = (np.pi) / segment_duration # Angular frequency
    # frequency gain, squared since R is a quadratic term
    return 1. / ((d * omega**order)**2) 

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def debug_mat(m):
    plt.cla()
    plt.figure(figsize=(5,5))
    m = (np.abs(m) > 0).astype(float)
    plt.imshow(m) # / np.max(m))
    plt.axis('equal')
    plt.show()
    

class DynamicalSystem:
    def __init__(self, order, subd, dim=2, segment_duration=0.5, discretization='exact'):
        '''
        Abstracts a chain of integrators
        '''
        self.order = order
        self.dim = dim
        self.subd = subd
        self.segment_duration = segment_duration
        self.dt = segment_duration / subd
        self.discretization = discretization
        
        # (univariate) linear system definition x_dot = Ax + Bu
        A = np.zeros((order,order))
        A[0:order-1, 1:order] = np.eye(order-1)
        self.A = A
        B = np.zeros((order,1))
        B[-1,:] = 1.
        self.B = B
        C = np.zeros((1, order))
        C[:,0] = 1
        self.C = C
        
    def discretize(self, dt=None, discretization='exact'):
        if dt is None:
            dt = self.dt
        
        Ad, Bd = discretize_system(self.A, self.B, dt, method=discretization) 
        # extend to include all dimension
        Ad = np.kron(Ad, np.eye(self.dim))
        Bd = np.kron(Bd, np.eye(self.dim))
        return Ad, Bd
    

def num_samples(subd, m):
    return (m-1)*subd+1

def solve_lqt_iterative(sys, reference, d, r=None):
    ''' Performs iterative discrete LQR with a finite horizon
        Input:
            A: system dynamics matrix (continuous, 1d version)
            B: system control matrix (continuous, 1d version)
            dt: time step
            r: coefficent for control cost 
            x0_: initial state of the system (full or position only)
            MuQ: block matrix of Mu's for each time step
            Q: block diagonal matrix with precision matrices for each time step
            discretzation_method: defaults to Zero Order Hold ('zoh'),
            full_output: if True optional output parameters will be returned
        based on:
        @article{Calinon16JIST,
          author="Calinon, S.",
          title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
          journal="Intelligent Service Robotics",
              publisher="Springer Berlin Heidelberg",
              doi="10.1007/s11370-015-0187-9",
              year="2016",
              volume="9",
              number="1",
              pages="1--29"
        }
    '''
    def mul(*args):
        ''' TODO remove Multiply an array of matrices'''
        X = args
        if len(X)==1:
            X = X[0]

        x = X[0];
        for y in X[1:]:
            x = np.dot(x,y)
        return x

    scale = 1.0
    k = sys.order    
    dim = sys.dim
    dt = sys.dt
    
    cDim = dim*k # dimension of the full state
    
    MuQ, Q, K = reference
    n = MuQ.shape[1] #// cDim
    #dt = 1./(n-1)
    
    if r is None:
        r = SHM_r(d, k, sys.segment_duration)
    
    A, B = sys.discretize()

    # In this case we keep the column layout
    Xi_hat = MuQ 
    # print(Xi_hat.shape, Q.shape, A.shape, n)
    # Sensor matrix (assuming only position is observed) 
    # Actually unused here and directly setting unobserved Q entries to 0 weight
    C = np.kron(np.hstack([np.ones((1,1)), np.zeros((1, k-1))]),
              np.eye(dim))
    # Control cost matrix
    R = np.eye(dim) * r

    P = np.zeros((cDim, cDim, n))
    i = n-1
    P[ :, :, -1] = Q[i*cDim:, i*cDim:]
    
    d = np.zeros((cDim, n))
    # Riccati recursion
    for i in range(n-2, 0, -1):
        # Not so fun to write this with NumPy
        Qi = Q[i*cDim:(i+1)*cDim, i*cDim:(i+1)*cDim]

        P[:,:,i] = Qi - mul( A.T, 
                        mul( P[:,:,i+1], B, inv( mul( B.T, P[:,:,i+1], B ) + R), B.T, P[:,:,i+1] ) - P[:,:,i+1], A)

        d[:,i] = ( mul( A.T - mul( A.T, P[:,:,i+1], B, inv(R + mul( B.T, P[:,:,i+1], B ) ), B.T ), 
                    mul( P[:,:,i+1], mul( A, Xi_hat[:, i] ) - Xi_hat[:, i+1] ) + d[:,i+1]) )

    x0 = Xi_hat[:,0].reshape(-1,1) 
    Y = np.zeros((cDim, n))
    x = x0[:,0]
    U = np.zeros((dim, n))

    for i in range(n):
        Y[:,i] = x
        p = P[:, :, i]
        mu = Xi_hat[:, i]
        
        G = mul(np.linalg.pinv( mul(B.T, p, B) + R ), B.T )
        
        # feedback gain
        K = mul( G, p, A )
        # Feedforward term
        M = mul( -G, mul(p, mul(A, mu) - mu) + d[:,i] )
        # Control command (highest order derivative of system)
        u = mul( K, mu - x ) + M
        # New state
        x = mul(A, x) + mul(B, u)
        U[:,i] = u

    res = {}
    res['sys'] = sys
    res['uSigma'] = []
    res['xSigma'] = []
    res['Rq'] = None
    res['u'] = U
    res['x'] = Y
    res['x0'] = x0
    res['opt_x0'] = False
    return res


    if full_output:
        return Y[0:dim, :]/scale, U
    else:
        return Y[0:dim, :]/scale


def solve_lqt(sys, references, d, opt_x0=False, r=None):
    '''
    Batch solution to LQT problem
    '''
    k = sys.order    
    dim = sys.dim
    dt = sys.dt
    
    cDim = dim*k # dimension of the full state
    
    if type(references) != list:
        references = [references]
        
    MuQ, Q, K = references[0]
    n = MuQ.shape[0] // cDim
    #dt = 1./(n-1)
    
    if r is None:
        r = SHM_r(d, k, sys.segment_duration)
    
    R = np.eye(dim) * r
    R = np.kron(np.eye(n-1), R)
    
    Ad, Bd = sys.discretize()

    Qs = []
    Sxs = []
    Sus = []
    
    ## Build Sx and Su matrices for batch LQR, see Eq. (14, 15)
    Su = np.zeros((cDim*n, dim*(n-1)))
    Sx = np.kron(np.ones((n,1)),np.eye(cDim))
    M = np.array(Bd)
    for i in range(1, n):
        Sx[ i*cDim:n*cDim, : ] = Sx[ i*cDim:n*cDim, : ] @ Ad
        Su[ i*cDim:(i+1)*cDim, 0:i*dim ] = M
        M = np.hstack([ Ad @ M[:, 0:dim], M ])
    SuTQ = Su.T @ Q
    SxTQ = Sx.T @ Q

    Qs = []
    SxTQs = []
    SuTQs = []
    MuQs = []
    for MuQi, Qi, Ki in references:
        Qs.append(Qi)
        SxTQs.append(Sx.T @ Qi)
        SuTQs.append(Su.T @ Qi)
        MuQs.append(MuQi)
    def sum_prod(elems):
        return np.sum(elems, axis=0)
    
    pa = 0 
    pb = n-1 

    # Solution
    Uu = sum_prod([SuTQi @ Su + R for SuTQi in SuTQs]) # U coefficients
    Uxbar = sum_prod([SuTQi @ MuQi for SuTQi, MuQi in zip(SuTQs, MuQs)]) # U dependent
    X0bar = sum_prod([SxTQi @ MuQi for SxTQi, MuQi in zip(SxTQs, MuQs)]) 

    invf = inv #pinv
    # def invf(M):
    #     c = inv(np.linalg.cholesky(M))
    #     return np.dot(c.T,c)

    #pdb.set_trace()

    if K is None:
        if opt_x0: # Optimise for initial state
            
            Ux0 = sum_prod([SuTQi @ Sx for SuTQi in SuTQs])
            X0u = sum_prod([SxTQi @ Su for SxTQi in SxTQs])
            X0x0 = sum_prod([SxTQi @ Sx for SxTQi in SxTQs])
            
            Rq = np.block([[Uu,  Ux0 ],
                           [X0u, X0x0]])
            rq = np.vstack([Uxbar, X0bar])
            ux0 = invf(Rq) @ rq #np.linalg.solve(Rq, rq)
            uhat = ux0
            u = ux0[:Uxbar.shape[0]]
            x0 = ux0[Uxbar.shape[0]:Uxbar.shape[0]+cDim]
            
            uSigma = invf(Rq) #; % + eye((nbData-1)*nbVarU) * 1E-10;
            utox = np.hstack([Su, Sx])
            xSigma = utox @ uSigma @ utox.T
        else: # Fixed initial state
            
            x0 = MuQ[:cDim].reshape(-1,1)
            Rq = sum_prod([SuTQi @ Su + R for SuTQi in SuTQs])
            rq = sum_prod([SuTQi @ (MuQi - Sx@x0) for SuTQi, MuQi in zip(SuTQs, MuQs)])

            u = invf(Rq) @ rq #solve(Rq, rq) #invf(Rq) @ rq #np.linalg.solve(Rq, rq)
            uhat = u
        
            uSigma = invf(Rq) #; % + eye((nbData-1)*nbVarU) * 1E-10;
            utox = Su #np.hstack([Su, Sx])
            xSigma = utox @ uSigma @ utox.T
    else: # Periodic solution

        opt_x0 = True

        Ux0 = SuTQ @ Sx
        Ul = Su.T @ K.T

        X0u = SxTQ @ Su
        X0x0 = SxTQ @ Sx
        X0l = Sx.T @ K.T
                        
        Lu = K@Su
        Lx0 = K@Sx
        Lxl = np.zeros((cDim, cDim))
        
        Lxl = np.zeros((cDim, cDim))

        X0bar = SxTQ @ MuQ

        Rq = np.block([[Uu,  Ux0,  Ul ],
                        [X0u, X0x0, X0l],
                        [Lu,  Lx0,  Lxl]])
        rq = np.vstack([Uxbar, X0bar,  np.zeros((cDim, 1))])
        ux0 = invf(Rq) @ rq #solve(Rq, rq) # #np.linalg.solve(Rq, rq)
        uhat = ux0
        u = ux0[:Uxbar.shape[0]]
        x0 = ux0[Uxbar.shape[0]:Uxbar.shape[0]+cDim]
        
        uSigma = invf(Rq) #; % + eye((nbData-1)*nbVarU) * 1E-10;
        utox = np.hstack([Su, Sx, K.T])
        xSigma = utox @ uSigma @ utox.T    

    res = {}

    y = Sx @ x0 + Su @ u
    x = np.reshape(y, (n, cDim)).T 
    u = np.reshape( u, (n-1, dim) ).T # control command     

    res['sys'] = sys
    res['uSigma'] = uSigma
    res['xSigma'] = xSigma
    res['Rq'] = Rq
    res['u'] = u
    res['x'] = x
    res['x0'] = x0
    res['opt_x0'] = opt_x0
    res['uhat'] = uhat # Raw result from optimisation
    return res

def reshape_stacked_reference(sys, X):
    cDim = sys.dim * sys.order
    n = X.shape[0] // cDim
    return np.reshape(X, (n, cDim)).T 

def get_Q_entries(sys, Q):
    cDim = sys.dim * sys.order
    n = Q.shape[0] // cDim
    return [Q[i*cDim:i*cDim+cDim, i*cDim:i*cDim+cDim] for i in range(n)]

def stochastic_sample_x(res, n_samples, sigma, n_eigs=-1):
    ''' Generate samples from the trajectory distribution'''
    samps = []
    D, V  =  largest_eigs(res['xSigma'], n_eigs) 
    n_eigs = D.shape[0]
    D = np.real(np.diag(D))
    V = np.real(V)
    x = res['x']
    for i in range(n_samples):
         o = np.random.randn(n_eigs, 1)*sigma
         offset =  V @ D**.5 @ o
         O = np.reshape( offset, (x.shape[1], x.shape[0])).T
         samp = x + O
         samps.append(samp)

    return samps

def stochastic_sample_u(res, subd, n_samples, sigma, n_eigs=-1):
    ''' Generate samples from the trajectory distribution'''
    samps = []
    
    D, V  =  largest_eigs(res['uSigma'], n_eigs) 
    n_eigs = D.shape[0]
    D = np.real(np.diag(D))
    V = np.real(V)
    
    # Prepare 
    x = res['x']
    u = res['u']
    x0 = res['x0']
    uhat = res['uhat']
    opt_x0 = res['opt_x0']
    sys = res['sys']
    #discretization = res['discretization']
    
    n = x.shape[1]
    dim = u.shape[0]
    cDim = x.shape[0]

    dt = sys.dt #1. / n
    dts = dt/subd

    ns = num_samples(subd, n) #(n-1)*subd+1
    xs = np.zeros((cDim, ns))
    us = np.zeros((2, ns-1))

    #A, B = res['A'], res['B']
    Ad, Bd = sys.discretize(dts)
    # extend to include all dimension
    samples = []

    for _ in range(n_samples):
        xs = np.zeros((cDim, ns))
        us = np.zeros((2, ns-1))

        o = np.random.randn(n_eigs, 1)*sigma
        offset =  V @ D**.5 @ o
        uh = uhat + offset

        ix0 = u.shape[0]*u.shape[1]
        u = uh[:ix0]
        if opt_x0:
            x0 = uh[ix0:ix0+cDim]
        u = np.reshape( u, (n-1, dim) ).T

        t = 0
        xs[:,0] = x0[:,0]
        #pdb.set_trace()
        for i in range(n-1):
            for j in range(subd):
                xs[:,t+1] = Ad @ xs[:,t] + Bd @ u[:,i]
                us[:,t] = u[:,i]
                t += 1

        samples.append(xs)
    return samples

def sample_spline(res, subd):
    x = res['x']
    sys = res['sys']
    
    n = x.shape[1]
    ns = num_samples(subd, n) #(n-1)*subd+1
    
    from scipy.interpolate import CubicSpline
    ts = np.linspace(0, 1, x.shape[1])
    yspline = CubicSpline(ts, x.T)
    yy = yspline(np.linspace(0, 1, ns)).T
    return yy
    
def sample_trajectory(res, subd=1, observe=False, get_commands=False, discretization='exact'):
    '''Resample a possibly sparse trajectory'''
    x = res['x']
    u = res['u']
    x0 = res['x0']
    sys = res['sys']
    
    n = x.shape[1]
    dim = u.shape[0]
    cDim = x.shape[0]
    
    dt = sys.dt #1. / n
    dts = dt/(subd)

    ns = num_samples(subd, n) #(n-1)*subd+1
    xs = np.zeros((cDim, ns))
    us = np.zeros((dim, ns-1))

    Ad, Bd = sys.discretize(dts) #discretize_system(A, B, dts, method=discretization) #'zoh') #method='euler')

    #print(np.min(Ad[np.where(Ad>0)]))
    t = 0
    xs[:,0] = x0[:,0]
    #pdb.set_trace()
    for i in range(n-1):
        # xs[:,t] = x[:,i]
        for j in range(subd):
            uu = u[:,i]
            
            #Add, Bdd = sys.discretize(sys.dt*(j/subd))
            #xs[:,t] = Add @ x[:,i] + Bdd @ uu # uu[:,i]
            xs[:,t+1] = Ad @ xs[:,t] + Bd @ uu #u[:,i]
            #xs[:,t+1] = Ad @ xs[:,t] + Bd @ uu #u[:,i]
            
            us[:,t] = uu #u[:,i]
            t += 1
    #xs[:,-1] = x[:,-1] #x0[:,0]
    x = xs
    u = us
    if observe:
        x = x[:dim,:]
    if get_commands:
        return x, u
    return x


def rbf(x, mu, sigma):
    return np.exp(-((x - mu)**2)/(2*sigma**2))

def chord_lengths( P, closed=0 ):
    ''' Chord lengths for each segment of a contour '''
    if closed:
        P = np.c_[P, P[:,0]]
    D = np.diff(P, axis=1)
    L = np.sqrt( D[0,:]**2 + D[1,:]**2 )
    return L
#endf

def cum_chord_lengths( P, closed=0 ):
    ''' Cumulative chord lengths '''
    if len(P.shape)!=2:
        return []
    if P.shape[1] == 1:
        return np.zeros(1)
    L = chord_lengths(P, closed)
    return np.cumsum(np.concatenate([[0.0],L]))
#endf

from scipy.special import erf

def centripetal_parameters(P, m, n, alpha):
    L = chord_lengths(P)
    L = L**alpha
    L = np.cumsum(np.concatenate([[0.0],L]))

    L = L / L[-1]

    if np.any(np.isnan(L)):
        pdb.set_trace()
        
    L[-1] = 1. # making sure for prec. errors
    return L*(n-1)

def subdivide(P):
    P = P.T
    D = np.diff(P, axis=0)
    P2 = [P[0]]
    for d in D:
        P2.append(P2[-1]+d*0.5)
        P2.append(P2[-1]+d*0.5)
    return np.array(P2).T

def nielson_foley_metric(P, affine=True):
    if not affine:
        return np.eye(P.shape[0])

    mu = np.mean(P, axis=1)
    Sigma = np.zeros((2,2))
    n = P.shape[1]
    V = P - mu.reshape(-1,1)
    Sigma = (V @ V.T)/n
    if det(Sigma)<1e-6:
        pdb.set_trace()
        return np.eye(P.shape[0])
    G = inv(Sigma)
    return G

def nielson_foley_norm(d, G):
    d = d.reshape(-1,1)
    l =  d.T @ G @ d 
    return np.sqrt(l[0,0])

def nielson_foley_distance(p, q, G):
    return nielson_foley_norm(p - q, G)

def angle_between(a, b):
    return np.arctan2( a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1] )


# cagd_lib (github) has similar code
def nielson_foley_parameterization(P, factor=1.5, affine=False):
    G = nielson_foley_metric(P, affine=affine)
    D = np.diff(P, axis=1)
    n = P.shape[1]
    d = np.zeros(n)
    for i in range(n-1):
        d[i] = nielson_foley_norm(D[:,i], G)


    theta = np.zeros(n)
    for i in range(1, n-1):
        dbase = nielson_foley_distance(P[:,i-1], P[:,i+1], G)
        #pdb.set_trace()
        v = ((d[i-1]**2 + d[i]**2 - dbase**2) /
                               (2*d[i]*d[i-1]) )
        v = np.clip(v, -1, 1)
        alpha = (np.pi - np.arccos(v))
        theta[i] = min(alpha, np.pi/2)

        #theta[i] = abs(angle_between(D[:,i-1], D[:,i]))

    #pdb.set_trace()
    h = np.zeros(n-1)
    for i in range(0, n-1):
        a = 1.
        if i > 0:
            a += factor*((theta[i]*d[i-1]) / (d[i-1] + d[i]))
        a += factor*((theta[i+1]*d[i+1]) / (d[i] + d[i+1]))
        h[i] = d[i]*a

    h = np.cumsum(np.concatenate([[0], h]))
    h = h / h[-1]
    h[-1] = 1.
    return h


def nielson_foley_parameters(P, n, factor=1.5):
    h = nielson_foley_parameterization(P, factor)
    return h * (n-1)

def fang_parameterization(P):
    #pdb.set_trace()

    G = nielson_foley_metric(P)

    L = chord_lengths(P)
    Lc = L**0.5


    D = np.diff(P, axis=1).T
    m = P.shape[1]
    theta = []
    l = []
    theta = []

    # for i in range(1,m-1):
    #     l.append(nielson_foley_distance(P[:,i-1], P[:,i+1], G))
    #     #l.append(norm(P[:,i-1]-P[:,i+1]))
    #     #pdb.set_trace()
    #     v = ((d[i-1]**2 + d[i]**2 - l[-1]**2) /
    #                             (2*d[i]*d[i-1]) )
    #     v = np.clip(v, -1, 1)
    #     alpha = (np.pi - np.arccos(v))
    #     theta.append(min(alpha, np.pi/2))
    #     #theta.append(angle_between(D[i-1], D[i]))
    # Lc = d #**0.5

    for i in range(m-2):
        # theta.append(abs(angle_between(D[i], D[i+1])))
        # sides = [norm(P[:,i] - P[:,i+2]),
        #           norm(D[i]),
        #           norm(D[i+1])]
        # l.append(sides[0]) #np.min(sides))

        theta.append(abs(angle_between(D[i], D[i+1])))
        sides = [ norm(D[i]),
                  norm(D[i+1])]
        #s = np.min(sides)
        #l.append(
        l.append(np.min(sides))

    for i in range(1, m-1):
        e= 0
        a = 0.01 #0.01 # 0.0001 #0.1
        pw = 1 #1/3
        #dth = 2
        #pdb.set_trace()
        #if i < mt:
        #    Lc[i] += (a* ( (theta[i]*l[i])/(2*np.sin(theta[i]/2) +1e-5) ))**(pw) #(1 + np.sin(theta[i]/2))
        eps = 0.1
        #continue
        if i > 0 and i < m-2:
            e += (a*((theta[i-1]*l[i-1])/(2*np.cos(theta[i-1]/2) + eps)))**(pw)
        #else:
        Lc[i] += e

    Lc = np.cumsum(np.concatenate([[0.0],Lc]))
    #pdb.set_trace()
    Lc = Lc / Lc[-1]
    print('fong: ' + str(Lc))
    if np.any(np.isnan(L)):
        pdb.set_trace()
    return Lc


def fang_parameters(P, n):
    h = fang_parameterization(P)
    return h * (n-1)

def compute_parameters(P, n, alpha, psg=None):
    if psg is not None:
        return psg
    if alpha < 0.:
        if alpha < -5:
            tau = fang_parameters(P, n)
        else:
            tau = nielson_foley_parameters(P, n, -alpha)
    else:
        tau = centripetal_parameters(P, P.shape[1], n, alpha) # np.linspace(0, n-1, m)
    return tau

def interior_angle(a, b):
    th = np.arccos(np.clip(np.dot(a,b) / np.sqrt(np.dot(a, a) * np.dot(b, b)), -1, 1))
    if th > np.pi / 2:
        th = np.pi - th
    return th

def sigma_rbf(subd, activation_sigma):
    # Activation function for variable support tracking reference
    # (half) Full width at half maximum of Gaussian
    return max(1e-5, (subd*0.5*activation_sigma)/3)

def reference_activations(P, m, n, activation_sigma, alpha=0., get_activations=False, psg=None, get_ref=False):
    subd = (n - 1)//(m - 1)
    #print((n-1)/(m-1))
    sigma = sigma_rbf(subd, activation_sigma)
    
    tau = compute_parameters(P, n, alpha, psg=psg)
    
    # activations
    H = np.zeros((m, n))
    
    t = np.linspace(0, n-1, n)

    for i, tau_i in enumerate(tau):
        p = rbf(t, int(tau_i), sigma) #np.random.uniform(2, 130)) #(n/m)*0.01)
        #plt.plot(t, p, plut.default_color(i)) #*200)
        #plt.fill_between(t, 0, p, where=p>0, facecolor=plut.default_color(i), alpha=0.3)
        H[i,:] = p

    P = H
    H = H / (np.tile(np.sum(H, axis=0) + np.exp(-6), (m, 1)))
    
    I = []
    T = []
    v = np.ones(n)*-1
    #pdb.set_trace()
    for t in range(n):
        for i, h in enumerate(H):
            if h[t] > 0.5:
                I.append(i)
                T.append(t)
                v[t] = i
                break
    if get_ref:
        return v.astype(int)
    if get_activations:
        return P, H
    return I, T

#def stepwise_reference()
def stepwise(m, n):
    T = range(n)
    I = np.linspace(0, float(m)-0.1, n).astype(int)
    return I, T

def cartesian_reference(sys, P, Sigma, activation_sigma, periodic=False, iterative=False, stop_movement=True, alpha=0., psg=None, clamp=False, diag=0.):
    if type(Sigma)==list:
        Sigma = np.dstack(Sigma)
    if periodic:
        P = np.hstack([P, P[:,:1]])
        Sigma = np.dstack([Sigma, Sigma[:,:,:1]])
        stop_movement = False
    elif clamp:
        if clamp == 1:
            P = np.hstack([P[:,:1], P, P[:,-1:]])
            Sigma = np.dstack([Sigma[:,:,:1], Sigma, Sigma[:,:,-1:]])
        elif clamp == 2:
            P = np.hstack([P[:,:1], P])
            Sigma = np.dstack([Sigma[:,:,:1], Sigma])
        elif clamp == 3:
            P = np.hstack([P, P[:,-1:]])
            Sigma = np.dstack([Sigma, Sigma[:,:,-1:]])
        #stop_movement = False
        
    pdim = P.shape[0]
    dim = sys.dim
    #assert(dim==P.shape[0])
    order = sys.order
    segment_duration = sys.segment_duration
    subd = sys.subd
    
    cDim = dim*order
    m = P.shape[1]
    
    # number of samples:    
    n = num_samples(subd, m) #(m-1)*subd+1
        
    # full states
    Mu = np.vstack([P, np.zeros((cDim-pdim, m))])
    # precisions
    Lambda = []
    for i in range(m):
        l = np.eye(cDim)*diag
        if np.max(np.abs(Sigma[:,:,i])) > 1e-10:
            l[0:pdim,0:pdim] = np.linalg.inv(Sigma[:,:,i]) # + np.eye(pdim)*1e-5)
        Lambda.append(l)

    MuQ = [np.zeros(cDim) for i in range(n)]
    Q = [np.eye(cDim)*0. for i in range(n)]
    
    # state activations
    states, timesteps = reference_activations(P, m, n, activation_sigma, alpha, psg=psg)

    # optionally 
    tau = compute_parameters(P, n, alpha, psg=psg).astype(int)
    
    for i, t in zip(states, timesteps):
        MuQ[t] = Mu[:,i]
        Q[t] = Lambda[i]

    if iterative:
        Q = block_diag(*Q)
        if stop_movement:
            # print("STOP MOVEMENT")
            # TODO add constraint here
            # i = (n-1)*cDim
            # K = np.zeros((cDim, n*cDim))
            # K[:, i:i+cDim] = np.diag(np.concatenate([np.zeros(dim), np.ones(cDim-dim)]))
            # print('stopit')
            amt = 1
            i = (n-1)*cDim
            Q[i:i+cDim,i:i+cDim] = np.eye(cDim)*amt
            i = 0
            Q[i:i+cDim,i:i+cDim] = np.eye(cDim)*amt

        return np.array(MuQ).T, Q, [] #np.array(Q)
    
    MuQ = np.hstack(MuQ).reshape(-1,1)
    Q = block_diag(*Q)
    
    pa, pb = 0, n-1
    #import pdb
    #pdb.set_trace()
    if periodic:
        K = np.zeros((cDim, n*cDim))
        K[:, pa*cDim:pa*cDim+cDim] = np.eye(cDim)
        K[:, pb*cDim:pb*cDim+cDim] = -np.eye(cDim)
    else:
        K = None
        if stop_movement:
            #print("STOP MOVEMENT")
            # TODO add constraint here
            # i = (n-1)*cDim
            # K = np.zeros((cDim, n*cDim))
            # K[:, i:i+cDim] = np.diag(np.concatenate([np.zeros(dim), np.ones(cDim-dim)]))
            # print('stopit')
            i = (n-1)*cDim
            Q[i+dim:i+cDim,i+dim:i+cDim] = np.eye(cDim-dim)
            Q[i:i+dim,i:i+dim] = np.eye(dim)*1 #100

        
    return MuQ, Q, K

def sparse_reference(sys, P, Sigma, activation_sigma, periodic=False, stop_movement=True, alpha=0., psg=None, clamp=False, diag=0.):
    if periodic:
        P = P + [P[0]]
        Sigma = Sigma + [Sigma[0]]
        stop_movement = False

    dim = sys.dim
    #assert(dim==P.shape[0])
    order = sys.order
    segment_duration = sys.segment_duration
    subd = sys.subd

    cDim = dim*order
    m = len(P)

    # number of samples:
    n = num_samples(subd, m) #(m-1)*subd+1

    # full states
    Mu = np.zeros((cDim, m))
    for i, p in enumerate(P):
        Mu[:p.shape[0],i] = p

    # precisions
    Lambda = []
    for i in range(m):
        l = np.eye(cDim)*diag
        pdim = Sigma[i].shape[0]
        if np.max(np.abs(Sigma[i])) > 1e-10:
            l[0:pdim,0:pdim] = np.linalg.inv(Sigma[i]) # + np.eye(pdim)*1e-5)
        Lambda.append(l)

    MuQ = [np.zeros(cDim) for i in range(n)]
    Q = [np.eye(cDim)*0. for i in range(n)]

    # state activations
    P = np.array([p[:2] for p in P]).T
    states, timesteps = reference_activations(P, m, n, activation_sigma, alpha, psg=psg)

    # optionally
    tau = compute_parameters(P, n, alpha, psg=psg).astype(int)

    for i, t in zip(states, timesteps):
        MuQ[t] = Mu[:,i]
        Q[t] = Lambda[i]

    MuQ = np.hstack(MuQ).reshape(-1,1)
    Q = block_diag(*Q)

    pa, pb = 0, n-1
    #import pdb
    #pdb.set_trace()
    if periodic:
        K = np.zeros((cDim, n*cDim))
        K[:, pa*cDim:pa*cDim+cDim] = np.eye(cDim)
        K[:, pb*cDim:pb*cDim+cDim] = -np.eye(cDim)
    else:
        K = None
        if stop_movement:
            # print("STOP MOVEMENT")
            # TODO add constraint here
            # i = (n-1)*cDim
            # K = np.zeros((cDim, n*cDim))
            # K[:, i:i+cDim] = np.diag(np.concatenate([np.zeros(dim), np.ones(cDim-dim)]))
            # print('stopit')
            i = (n-1)*cDim
            #Q[i+dim:i+cDim,i+dim:i+cDim] = np.eye(cDim-2)
            Q[i:i+dim,i:i+dim] = np.eye(dim)*100000


    return MuQ, Q, K


def semitied_lqt(P, **kwargs):
    params = dict(angle=0,
                   isotropy=0.5,
                   sigma=70,
                   activation_sigma=0.8,
                   d=3.,
                   order=4,
                   segment_duration=0.5,
                   opt_x0=False,
                   stop_movement=True,
                   periodic=False ,
                   subd=10,
                   subsample=2,
                   dim=2,
                   variability=[],
                   )
    params.update(**kwargs)
    dim = params['dim']
    Sigma = []
    cov = np.eye(dim)*(params['sigma']**2)
    cov[:2,:2] = make_sigma(radians(params['angle']), [params['isotropy']*params['sigma'], params['sigma']])

    for p in P.T:
        Sigma.append(cov)
    if params['variability']:
        # print(params['variability'])
        Sigma = [w*s for w, s in zip(params['variability'], Sigma)]

    Sigma = np.array(Sigma).T
    sys = DynamicalSystem(params['order'], params['subd'], segment_duration=params['segment_duration'], dim=params['dim'])
    ref = cartesian_reference(sys, P, Sigma,
                                    activation_sigma=params['activation_sigma'],
                                    stop_movement=params['stop_movement'], periodic=params['periodic'])
    res = solve_lqt(sys, ref, params['d'], opt_x0=params['opt_x0']) #, r=1e-43)
    if params['subsample'] > 0:
        x = sample_trajectory(res, params['subsample'])
    else:
        x = res['x']
    return x


def num_bezier(n_ctrl, degree=3):
    if type(n_ctrl) == np.ndarray:
        n_ctrl = len(n_ctrl)
    return int((n_ctrl - 1) / degree)


