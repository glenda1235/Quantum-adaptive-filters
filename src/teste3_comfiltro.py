import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# Parâmetros
alpha = 1.0
n_th = 0.5
eta = 0.8
dim = 20
threshold = 1e-2  # Limiar para filtragem (ajustável)

# Estado coerente
coherent = coherent_dm(dim, alpha)

# Ruído térmico
thermal = thermal_dm(dim, n_th)

# Estado misto (com perda)
mixed_state = eta * coherent + (1 - eta) * thermal

# Operador quadratura X
x = (destroy(dim) + destroy(dim).dag()) / np.sqrt(2)

# Função SNR
def compute_snr(state, observable):
    mean = expect(observable, state)
    var = expect(observable**2, state) - mean**2
    return 10 * np.log10(mean**2 / var)

# Filtro adaptativo: elimina componentes com baixa probabilidade
def adaptive_filter(state, threshold):
    filtered_data = state.full().copy()
    diag = np.real(np.diag(filtered_data))
    for i in range(len(diag)):
        if diag[i] < threshold:
            filtered_data[i, :] = 0
            filtered_data[:, i] = 0
    # Normaliza novamente
    filtered_data /= np.trace(filtered_data)
    return Qobj(filtered_data, dims=state.dims)

# Aplica o filtro adaptativo
filtered_state = adaptive_filter(mixed_state, threshold)

# SNRs
snr_before = compute_snr(mixed_state, x)
snr_after = compute_snr(filtered_state, x)

print(f"SNR antes do filtro: {snr_before:.2f} dB")
print(f"SNR depois do filtro: {snr_after:.2f} dB")
