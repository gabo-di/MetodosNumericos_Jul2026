# %% [markdown]
# # Dia 3: Scientific Machine Learning
#
# ## Objetivos
#
# - Usar PyTorch para aprender una correccion a un modelo fisico
#
# - Integrar una ODE diferenciable con `torchdiffeq`
#
# - Entrenar una red neuronal usando error de trayectoria
#
# - Comparar:
#
# $$
# y_{true} = \text{pendulo no lineal}
# $$
#
# contra
#
# $$
# y_{hat} = \text{oscilador armonico}\times(1+\text{correccion})
# $$


# %% [markdown]
# # Problema fisico: pendulo no lineal
#
# El pendulo real satisface:
#
# $$
# \frac{d^2\theta}{dt^2}
# =
# -\omega_0^2\sin\theta.
# $$
#
# Para angulos pequenos:
#
# $$
# \sin\theta \approx \theta.
# $$
#
# Entonces:
#
# $$
# \frac{d^2\theta}{dt^2}
# =
# -\omega_0^2\theta.
# $$
#
# Queremos aprender una correccion:
#
# $$
# \frac{d^2\theta}{dt^2}
# =
# -\omega_0^2\theta
# \left[
# 1 + F_\alpha(\theta)
# \right].
# $$
#
# donde:
#
# $$
# y = (\theta, v), \qquad v = \dot\theta.
# $$


# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special

import torch
import torch.nn as nn
import torch.optim as optim

from torchdiffeq import odeint


torch.set_num_threads(1)

np.random.seed(42)
torch.manual_seed(42)

torch.set_default_dtype(torch.float32)

omega0 = 1.0


# %% [markdown]
# # Periodo del pendulo real
#
# El periodo del oscilador armonico es:
#
# $$
# T_0 = \frac{2\pi}{\omega_0}.
# $$
#
# Pero el pendulo real tiene periodo dependiente de la amplitud:
#
# $$
# T(\theta_0)
# =
# \frac{4}{\omega_0}
# K\left(\sin^2\frac{\theta_0}{2}\right),
# $$
#
# donde $K$ es la integral eliptica completa de primera especie.
#
# Usaremos este periodo para entrenar sobre una oscilacion completa.


# %%
def periodo_pendulo(theta0, omega0):
    k2 = np.sin(theta0 / 2.0)**2
    return 4.0 * special.ellipk(k2) / omega0


theta0_values = np.linspace(0.1, 2.6, 20)

T_real = np.array([periodo_pendulo(theta0, omega0) for theta0 in theta0_values])
T_arm = 2 * np.pi / omega0 * np.ones_like(T_real)

plt.plot(theta0_values, T_real, "o-", label="pendulo real")
plt.plot(theta0_values, T_arm, "--", label="armonico")
plt.xlabel(r"amplitud inicial $\theta_0$")
plt.ylabel("periodo")
plt.title("Periodo vs amplitud")
plt.legend()
plt.show()


# %% [markdown]
# # Modelos diferenciales
#
# Usamos el estado:
#
# $$
# y = (\theta, v).
# $$
#
# El pendulo real es:
#
# $$
# \dot\theta = v,
# \qquad
# \dot v = -\omega_0^2\sin\theta.
# $$
#
# El oscilador armonico es:
#
# $$
# \dot\theta = v,
# \qquad
# \dot v = -\omega_0^2\theta.
# $$


# %%
class TruePendulum(nn.Module):
    def __init__(self, omega0):
        super().__init__()
        self.omega0 = omega0

    def forward(self, t, y):
        theta = y[:, 0:1]
        v = y[:, 1:2]

        dtheta = v
        dv = -self.omega0**2 * torch.sin(theta)

        return torch.cat([dtheta, dv], dim=1)


class HarmonicPendulum(nn.Module):
    def __init__(self, omega0):
        super().__init__()
        self.omega0 = omega0

    def forward(self, t, y):
        theta = y[:, 0:1]
        v = y[:, 1:2]

        dtheta = v
        dv = -self.omega0**2 * theta

        return torch.cat([dtheta, dv], dim=1)

pendulo_true = TruePendulum(omega0)
pendulo_armonico = HarmonicPendulum(omega0)


# %% [markdown]
# # Red neuronal de correccion
#
# La red recibe:
#
# $$
# y = (\theta, v)
# $$
#
# y devuelve:
#
# $$
# F_\alpha(\theta).
# $$
#
# Entonces el modelo aprendido es:
#
# $$
# \dot\theta = v,
# $$
#
# $$
# \dot v =
# -\omega_0^2\theta
# \left[
# 1 + F_\alpha(\theta)
# \right].
# $$


# %%
class CorrectedOscillator(nn.Module):
    def __init__(self, omega0):
        super().__init__()

        self.omega0 = omega0

        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Al inicio F = 0, entonces el modelo empieza como armonico.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def correction(self, y):
        theta = y[:, 0:1]

        theta_scale = np.pi

        y_scaled = torch.cat(
            [theta / theta_scale], # B, 1 -> B, 1
            dim=1
        )

        F = torch.tanh(self.net(y_scaled))

        return F

    def forward(self, t, y):
        theta = y[:, 0:1]
        v = y[:, 1:2]

        F = self.correction(y)

        dtheta = v
        dv = -self.omega0**2 * theta * (1.0 + F)

        return torch.cat([dtheta, dv], dim=1)


modelo = CorrectedOscillator(omega0)


# %% [markdown]
# # Datos de entrenamiento
#
# Entrenamos con varias amplitudes iniciales:
#
# $$
# \theta(0)=\theta_0,
# \qquad
# v(0)=0.
# $$
#
# Para cada amplitud entrenamos sobre un periodo real:
#
# $$
# t \in [0,T(\theta_0)].
# $$


# %%
theta0_train = [0.3, 0.8, 1.3, 1.8, 2.5]

n_time = 120

datos_train = []

for theta0 in theta0_train:
    T = periodo_pendulo(theta0, omega0)

    t = torch.linspace(0.0, T, n_time)
    dt = float(t[1] - t[0])

    y0 = torch.tensor([[theta0, 0.0]])

    with torch.no_grad():
        y_true = odeint(
            pendulo_true,
            y0,
            t,
            method="rk4",
            options={"step_size": dt}
        )

        y_arm = odeint(
            pendulo_armonico,
            y0,
            t,
            method="rk4",
            options={"step_size": dt}
        )

    datos_train.append(
        {
            "theta0": theta0,
            "T": T,
            "t": t,
            "dt": dt,
            "y0": y0,
            "y_true": y_true.detach(),
            "y_arm": y_arm.detach()
        }
    )

    print("theta0:", theta0, "T:", T)


# %% [markdown]
# # Antes de entrenar
#
# Comparamos:
#
# - pendulo real
# - oscilador armonico
#
# para una amplitud grande.


# %%
ejemplo = datos_train[-1]

t_np = ejemplo["t"].numpy()
y_true_np = ejemplo["y_true"][:, 0, :].numpy()
y_arm_np = ejemplo["y_arm"][:, 0, :].numpy()

plt.plot(t_np, y_true_np[:, 0], label="pendulo real")
plt.plot(t_np, y_arm_np[:, 0], "--", label="armonico")
plt.xlabel("t")
plt.ylabel(r"$\theta(t)$")
plt.title(r"Antes de entrenar: $\theta_0=2.3$")
plt.legend()
plt.show()


# %%
plt.plot(y_true_np[:, 0], y_true_np[:, 1], label="pendulo real")
plt.plot(y_arm_np[:, 0], y_arm_np[:, 1], "--", label="armonico")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$v$")
plt.title("Espacio de fase")
plt.legend()
plt.show()


# %% [markdown]
# # Loss function 
#
# Loss function compara la trayectoria integrada.
#
# Para cada amplitud:
#
# $$
# y_{true}(t) = \text{pendulo real},
# $$
#
# $$
# y_{hat}(t) = \text{modelo corregido}.
# $$
#
# Usamos:
#
# $$
# L =
# \left\langle
# \left(
# \frac{\theta_{hat}-\theta_{true}}{\theta_{scale}}
# \right)^2
# +
# \left(
# \frac{v_{hat}-v_{true}}{v_{scale}}
# \right)^2
# \right\rangle.
# $$
#
# El optimizador modifica los parametros de la red para reducir esta perdida.


# %%
theta_scale = np.pi
v_scale = 2.5


def trayectoria_loss(y_hat, y_true):
    theta_hat = y_hat[:, :, 0] # B, T, 2 -> B, T, theta
    v_hat = y_hat[:, :, 1] # B, T, 2 -> B, T, v

    theta_true = y_true[:, :, 0]
    v_true = y_true[:, :, 1]

    loss_theta = ((theta_hat - theta_true) / theta_scale)**2
    loss_v = ((v_hat - v_true) / v_scale)**2

    return torch.mean(loss_theta + loss_v)


# %% [markdown]
# # Entrenamiento
#
# Conceptos importantes:
#
# - `odeint`: integra la ODE diferenciable
# - `loss`: mide que tan mala es la trayectoria aprendida
# - `loss.backward()`: calcula gradientes
# - `optimizer.step()`: actualiza la red
# - `learning_rate`: controla el tamaño de cada actualizacion
# - `epoch`: una iteracion de entrenamiento


# %%
learning_rate = 2e-3
n_epochs = 700

optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)

loss_hist = []

for epoch in range(n_epochs):
    optimizer.zero_grad()

    loss = 0.0

    for data in datos_train:
        y_hat = odeint(
            modelo,
            data["y0"],
            data["t"],
            method="rk4",
            options={"step_size": data["dt"]}
        )

        loss = loss + trayectoria_loss(y_hat, data["y_true"])

    loss = loss / len(datos_train)

    loss.backward()
    optimizer.step()

    loss_hist.append(loss.item())

    if epoch % 100 == 0:
        print(epoch, loss.item())


# %%
plt.semilogy(loss_hist)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Entrenamiento")
plt.show()


# %% [markdown]
# # Comparacion despues de entrenar


# %%
theta0_test = 2.3
T_test = periodo_pendulo(theta0_test, omega0)

t_test = torch.linspace(0.0, T_test, 200)
dt_test = float(t_test[1] - t_test[0])

y0_test = torch.tensor([[theta0_test, 0.0]])

with torch.no_grad():
    y_true_test = odeint(
        pendulo_true,
        y0_test,
        t_test,
        method="rk4",
        options={"step_size": dt_test}
    )

    y_arm_test = odeint(
        pendulo_armonico,
        y0_test,
        t_test,
        method="rk4",
        options={"step_size": dt_test}
    )

    y_hat_test = odeint(
        modelo,
        y0_test,
        t_test,
        method="rk4",
        options={"step_size": dt_test}
    )


t_np = t_test.numpy()

y_true_np = y_true_test[:, 0, :].numpy()
y_arm_np = y_arm_test[:, 0, :].numpy()
y_hat_np = y_hat_test[:, 0, :].numpy()

plt.plot(t_np, y_true_np[:, 0], label="pendulo real")
plt.plot(t_np, y_arm_np[:, 0], "--", label="armonico")
plt.plot(t_np, y_hat_np[:, 0], ":", label="aprendido")
plt.xlabel("t")
plt.ylabel(r"$\theta(t)$")
plt.title(r"Comparacion de trayectorias")
plt.legend()
plt.show()


# %%
plt.plot(y_true_np[:, 0], y_true_np[:, 1], label="pendulo real")
plt.plot(y_arm_np[:, 0], y_arm_np[:, 1], "--", label="armonico")
plt.plot(y_hat_np[:, 0], y_hat_np[:, 1], ":", label="aprendido")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$v$")
plt.title("Espacio de fase")
plt.legend()
plt.show()


# %%
error_arm = np.sqrt(np.sum((y_arm_np - y_true_np)**2, axis=1))
error_hat = np.sqrt(np.sum((y_hat_np - y_true_np)**2, axis=1))

plt.semilogy(t_np, error_arm + 1e-30, label="armonico")
plt.semilogy(t_np, error_hat + 1e-30, label="aprendido")
plt.xlabel("t")
plt.ylabel("error de trayectoria")
plt.ylim(1e-7, 1e1)
plt.title("Error sobre un periodo")
plt.legend()
plt.show()


# %% [markdown]
# # Correccion aprendida
#
# En este ejemplo conocemos la correccion exacta.
#
# Queremos que:
#
# $$
# -\omega_0^2\sin\theta
# =
# -\omega_0^2\theta
# \left[
# 1 + F(\theta)
# \right].
# $$
#
# Entonces:
#
# $$
# 1 + F(\theta)
# =
# \frac{\sin\theta}{\theta}.
# $$
#
# Por lo tanto:
#
# $$
# F(\theta)
# =
# \frac{\sin\theta}{\theta} - 1.
# $$



# %%
theta_grid = np.linspace(-np.pi, np.pi, 400)
v_grid = np.zeros_like(theta_grid)

Y_grid = torch.tensor(
    np.stack([theta_grid, v_grid], axis=1),
    dtype=torch.float32
)

with torch.no_grad():
    F_learned = modelo.correction(Y_grid).numpy().reshape(-1)

F_true = np.zeros_like(theta_grid)

mask = np.abs(theta_grid) > 1e-8
F_true[mask] = np.sin(theta_grid[mask]) / theta_grid[mask] - 1.0
F_true[~mask] = 0.0

plt.plot(theta_grid, F_true, label="correccion exacta")
plt.plot(theta_grid, F_learned, "--", label="correccion aprendida")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$F(\theta, v=0)$")
plt.title("Correccion aprendida al modelo armonico")
plt.legend()
plt.show()

# %% [markdown]
# ## Out of distribution test

# %% 

theta0_test = 3.0
T_test = periodo_pendulo(theta0_test, omega0)

t_test = torch.linspace(0.0, T_test, 200)
dt_test = float(t_test[1] - t_test[0])

y0_test = torch.tensor([[theta0_test, 0.0]])

with torch.no_grad():
    y_true_test = odeint(
        pendulo_true,
        y0_test,
        t_test,
        method="rk4",
        options={"step_size": dt_test}
    )

    y_arm_test = odeint(
        pendulo_armonico,
        y0_test,
        t_test,
        method="rk4",
        options={"step_size": dt_test}
    )

    y_hat_test = odeint(
        modelo,
        y0_test,
        t_test,
        method="rk4",
        options={"step_size": dt_test}
    )


t_np = t_test.numpy()

y_true_np = y_true_test[:, 0, :].numpy()
y_arm_np = y_arm_test[:, 0, :].numpy()
y_hat_np = y_hat_test[:, 0, :].numpy()

plt.plot(t_np, y_true_np[:, 0], label="pendulo real")
plt.plot(t_np, y_arm_np[:, 0], "--", label="armonico")
plt.plot(t_np, y_hat_np[:, 0], ":", label="aprendido")
plt.xlabel("t")
plt.ylabel(r"$\theta(t)$")
plt.title(r"Comparacion de trayectorias")
plt.legend()
plt.show()


plt.plot(y_true_np[:, 0], y_true_np[:, 1], label="pendulo real")
plt.plot(y_arm_np[:, 0], y_arm_np[:, 1], "--", label="armonico")
plt.plot(y_hat_np[:, 0], y_hat_np[:, 1], ":", label="aprendido")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$v$")
plt.title("Espacio de fase")
plt.legend()
plt.show()


error_arm = np.sqrt(np.sum((y_arm_np - y_true_np)**2, axis=1))
error_hat = np.sqrt(np.sum((y_hat_np - y_true_np)**2, axis=1))

plt.semilogy(t_np, error_arm + 1e-30, label="armonico")
plt.semilogy(t_np, error_hat + 1e-30, label="aprendido")
plt.xlabel("t")
plt.ylabel("error de trayectoria")
plt.ylim(1e-7, 1e1)
plt.title("Error sobre un periodo")
plt.legend()
plt.show()

# %%
