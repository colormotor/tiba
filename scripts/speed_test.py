#%%
import numpy as np
from polygonsoup import plut, geom, utils
import matplotlib.pyplot as plt

Qs, dQs = utils.load_pkl('trajectory.pkl')
Qs = np.array(Qs)
dQs = np.array(dQs)

plut.figure((6, 4))
for i in range(7):
    dq = dQs[:, i]
    dq = np.diff(Qs[:,i])
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