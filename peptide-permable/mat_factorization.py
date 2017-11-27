import numpy as np
sim_matrix = 0*np.load('sim_matrix%s.npy.zip.npz' %0).items()[0][1]
if True:
    for start in range(0,7):
         sim_matrix =  sim_matrix + np.load('sim_matrix%s.npy.zip.npz' %start).items()[0][1]
V = 1-sim_matrix/100

die

W = 0.02*np.random.standard_normal(size=(len(V),5))
H = 0.02*np.random.standard_normal(size=(5,len(V)))
W = np.array(W,dtype=np.float64)
H = np.array(H,dtype=np.float64)
W[W<=0]=0
H[H<=0]=0

for i in range(0,100):
    H_update_n = np.matmul(W.T,V)
    H_update_d = np.matmul(np.matmul(W.T,W),H)
    W_update_n = np.matmul(V,H.T)
    W_update_d = np.matmul(np.matmul(W,H),H.T)
    W += 0.01*(W_update_n/W_update_d+0.0001)
    H += 0.01*(H_update_n/H_update_d+0.0001)
    W[W<=0]=0
    H[H<=0]=0
    print np.mean((V-np.matmul(W,H))**2)
    die
    


