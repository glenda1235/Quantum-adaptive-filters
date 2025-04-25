from qutip import coherent, thermal_dm
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
alpha = 1.0        # Amplitude do estado coerente
n_th = 0.5         # Número médio de fótons térmicos
eta = 0.8          # Transmissividade do canal
dim = 20           # Dimensão do espaço de Hilbert

# Estado coerente (ideal) como estado de densidade
coherent_state = coherent(dim, alpha) * coherent(dim, alpha).dag()

# Estado térmico (ruído)
thermal_state = thermal_dm(dim, n_th)

# Mistura dos estados (ruído no canal)
mixed_state = eta * coherent_state + (1 - eta) * thermal_state

# Filtro adaptativo simples (exemplo de filtro com threshold nas amplitudes)
def adaptive_filter(state, threshold):
    # Realiza a filtragem de elementos do estado abaixo do threshold
    state_data = state.full()  # Acessa os dados reais do estado
    state_data[np.abs(state_data) < threshold] = 0  # Zera os elementos abaixo do threshold
    return state.__class__(state_data)  # Cria um novo Qobj com os dados modificados

# Função para calcular o MSE (Erro Quadrático Médio)
def mse(state1, state2):
    # Calcula o erro quadrático médio entre dois estados
    return (state1 - state2).norm()**2

# Calcular MSE
thresholds = np.linspace(1e-4, 1e-2, 10)  # Vários thresholds
mse_values = []

for threshold in thresholds:
    filtered_state = adaptive_filter(mixed_state, threshold)
    mse_values.append(mse(coherent_state, filtered_state))

# Plotando o MSE ao longo de diferentes thresholds
plt.plot(thresholds, mse_values)
plt.xlabel('Threshold')
plt.ylabel('MSE')
plt.title('MSE vs Threshold')
plt.grid(True)
plt.show()

# Erro nas Quadraturas Estimadas (Exemplo)
def quadrature_error(state, coherent_state):
    # Considerando quadraturas X e P para um estado coerente
    X = (state + state.dag()).real
    P = 1j * (state - state.dag()).real
    X_coherent = (coherent_state + coherent_state.dag()).real
    P_coherent = 1j * (coherent_state - coherent_state.dag()).real

    # Erro nas quadraturas X e P
    error_X = np.abs(X - X_coherent).mean()
    error_P = np.abs(P - P_coherent).mean()

    return error_X, error_P

# Calcular erro nas quadraturas
error_X, error_P = quadrature_error(filtered_state, coherent_state)
print(f'Erro nas Quadraturas (X): {error_X}')
print(f'Erro nas Quadraturas (P): {error_P}')

# Plotando erro nas quadraturas
errors = [error_X, error_P]
labels = ['Erro em X', 'Erro em P']

plt.bar(labels, errors)
plt.ylabel('Erro')
plt.title('Erro nas Quadraturas Estimadas')
plt.show()
