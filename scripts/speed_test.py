#%%
import numpy as np
from polygonsoup import plut, geom, utils
import matplotlib.pyplot as plt

Qs, dQs = utils.load_pkl('trajectory.pkl')
Qs = np.array(Qs)
dQs = np.array(dQs)

def saturate_acceleration(dQ):
    ddQ = np.diff(dQ, axis=0)
    print(np.max(ddQ))
    print(np.mean(ddQ))
    sat = 0.1
    ddQ = np.clip(ddQ, -sat, sat)
    return np.cumsum(np.vstack([dQ[0], ddQ]), axis=0)
    return dQ
#dQs = saturate_acceleration(dQs)
Qss = geom.smoothing_spline(len(Qs), Qs, smooth_k=10)
dQss = geom.smoothing_spline(len(dQs), dQs, smooth_k=20)
#dQs = geom.smoothing_spline(len(dQs), dQs, smooth_k=10)
dt = 0.001
plut.figure((6, 4))
for i in range(7): #[0]: #range(1):
    dq = dQs[:, i]
    #plt.plot(dq)
    #dq = dQss[:, i]
    #q = np.diff(Qss[:,i])/dt
    plt.plot(dq)
plt.show()
# %%
def batch_indices(n, batch_size):
    n_batches = n//batch_size + 1
    if n_batches < 2:
        return [list(range(n))]
    I = np.linspace(0, n-1, n_batches).astype(int)
    return [list(range(a, b+1)) for a, b in zip(I, I[1:])]

print(batch_indices(100, 10))

#%%