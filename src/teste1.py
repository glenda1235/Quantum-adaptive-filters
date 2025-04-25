import numpy as np
import matplotlib.pyplot as plt
from qutip import *

#parametros do sistema
alpha = 1.0        #aplitude do estado coerente
n_th = 0.5         #numero medio de fotons termicos (????????)
eta = 0.8          #transmissividade do canal
dim = 20           #dimensão do espaço de hilbert (??????????)

#estado coerente (emissor - Alice)
state = coherent(dim, alpha)

#estado térmico (ruído)
thermal = thermal_dm(dim, n_th)

#canal: mistura estado com ruído térmico com perda (modelo simplificado)
mixed_state = eta * ket2dm(state) + (1 - eta) * thermal

#fidelidade entre o estado original e o estado após o canal
fid = fidelity(state, mixed_state)
print(f"Fidelidade entre os estados: {fid:.4f}")

#wigner function para visualização
x = np.linspace(-5, 5, 200)
W1 = wigner(state, x, x)
W2 = wigner(mixed_state, x, x)

#plotar
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
cont1 = axes[0].contourf(x, x, W1, 100, cmap='RdBu')
axes[0].set_title("Estado Coerente Original")
cont2 = axes[1].contourf(x, x, W2, 100, cmap='RdBu')
axes[1].set_title("Estado Após Canal com Ruído")
plt.tight_layout()
plt.show()