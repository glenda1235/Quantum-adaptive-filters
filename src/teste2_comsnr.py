import numpy as np
import matplotlib.pyplot as plt
from qutip import *

#Parâmetros
alpha = 1.0
n_th = 0.5
eta = 0.8
dim = 20

#Estado coerente (sinal original)
coherent = coherent_dm(dim, alpha)

#Ruído térmico (estado de fundo)
thermal = thermal_dm(dim, n_th)

#Canal com perda (simula ruído no canal)
mixed_state = eta * coherent + (1 - eta) * thermal

#Operadores quadratura (posição e momento)
x = (destroy(dim) + destroy(dim).dag()) / np.sqrt(2)
p = 1j * (destroy(dim).dag() - destroy(dim)) / np.sqrt(2)

#Função para calcular SNR com base na quadratura X
def compute_snr(state, observable):
    mean = expect(observable, state)
    var = expect(observable**2, state) - mean**2
    return 10 * np.log10(mean**2 / var)

#SNR antes (estado coerente puro)
snr_before = compute_snr(coherent, x)

#snr depois (estado ruidoso)
snr_after = compute_snr(mixed_state, x)

print(f"SNR antes (em dB): {snr_before:.2f}")
print(f"SNR depois (em dB): {snr_after:.2f}")
