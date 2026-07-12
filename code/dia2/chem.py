# %% [markdown]
# # Reacciones bioquimicas con stiff
#
# Ahora estudiamos un sistema de ecuaciones diferenciales que aparece en
# cinetica enzimatica.
#
# La reaccion es:
#
# $$
# E + S
# \underset{k_{-1}}{\overset{k_1}{\rightleftharpoons}}
# ES
# \overset{k_{cat}}{\longrightarrow}
# E + P.
# $$
#
# donde:
#
# - $E$ es la enzima libre
# - $S$ es el sustrato
# - $ES$ es el complejo enzima-sustrato
# - $P$ es el producto





# %% [markdown]
# ## Modelo completo: ley de accion de masas
#
# Definimos las velocidades:
#
# $$
# v_1 = k_1 [E] [S]
# $$
#
# $$
# v_{-1} = k_{-1} [ES]
# $$
#
# $$
# v_{cat} = k_{cat} [ES].
# $$
#
# Entonces:
#
# $$
# \frac{dS}{dt} = -v_1 + v_{-1}
# $$
#
# $$
# \frac{dE}{dt} = -v_1 + v_{-1} + v_{cat}
# $$
#
# $$
# \frac{dES}{dt} = v_1 - v_{-1} - v_{cat}
# $$
#
# $$
# \frac{dP}{dt} = v_{cat}
# $$




# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


# %%
def enzima_completo(t, y, k1, km1, kcat):
    S, E, ES, P = y

    ???

    return np.array([dSdt, dEdt, dESdt, dPdt])




# %% [markdown]
# ## Parametros
#
# Usamos parametros que producen dos escalas de tiempo:
#
# - union y disociacion rapidas
# - formacion de producto mas lenta
#
# Esto puede hacer que el sistema presente stiffness.




# %%
# parametros
k1 = 1e6     # union: E + S -> ES
km1 = 1e3    # disociacion: ES -> E + S
kcat = 10.0  # catalisis: ES -> E + P

# condiciones iniciales
S0 = 1e-4
E0 = 1e-6
ES0 = 0.0
P0 = 0.0

y0 = np.array([S0, E0, ES0, P0])

t_span = (0.0, 30.0)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = integrate.solve_ivp(
    enzima_completo,
    t_span,
    y0,
    args=(k1, km1, kcat),
    method="Radau",
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-12
)

print(sol.message)
print("success: ", sol.success)
print("status: ", sol.status)
print("nfev: ", sol.nfev)
print("njev: ", sol.njev)

S = sol.y[0]
E = sol.y[1]
ES = sol.y[2]
P = sol.y[3]

plt.plot(sol.t, S, label="S")
plt.plot(sol.t, P, label="P")
plt.xlabel("t")
plt.ylabel("concentracion")
plt.title("Sustrato y producto")
plt.legend()
plt.show()




# %%
plt.plot(sol.t, E, label="E")
plt.plot(sol.t, ES, label="ES")
plt.xlabel("t")
plt.ylabel("concentracion")
plt.title("Enzima libre y complejo")
plt.legend()
plt.show()




# %% [markdown]
# ## Cantidades conservadas
#
# La cantidad total de enzima debe conservarse:
#
# $$
# [E_{tot}] = [E] + [ES].
# $$
#
# Tambien se conserva la cantidad total de sustrato:
#
# $$
# [S_{tot}] = [S] + [ES] + [P].
# $$




# %%
E_total = E + ES
S_total = S + ES + P

print("E_total inicial: ", E_total[0])
print("E_total final: ", E_total[-1])
print("S_total inicial: ", S_total[0])
print("S_total final: ", S_total[-1])



# %%
plt.plot(sol.t, E_total - E_total[0], label=r"$E + ES$")
plt.plot(sol.t, S_total - S_total[0], label=r"$S + ES + P$")
plt.xlabel("t")
plt.ylabel("error absoluto")
plt.title("Conservacion")
plt.legend()
plt.show()


# %% [markdown]
# # Comparacion de solvers
#
# Un sistema con stiffness tiene escalas de tiempo muy diferentes.
#
# En este caso, el complejo $ES$ puede cambiar rapidamente al principio,
# mientras que $S$ y $P$ cambian mas lentamente.
#
# Comparamos:
#
# - `RK45`: metodo explicito general
# - `Radau`: metodo implicito para sistemas con stiffness



# %%
metodos = ["RK45", "Radau"]
soluciones = {}

for metodo in metodos:
    sol_m = integrate.solve_ivp(
        enzima_completo,
        t_span,
        y0,
        args=(k1, km1, kcat),
        method=metodo,
        rtol=1e-6,
        atol=1e-10
    )

    soluciones[metodo] = sol_m

    print(metodo)
    print("  success:", sol_m.success)
    print("  nfev:", sol_m.nfev)
    print("  njev:", sol_m.njev)
    print("  final P:", sol_m.y[3, -1])
    print()




# %%
for metodo in metodos:
    sol_m = soluciones[metodo]
    plt.scatter(sol_m.t, sol_m.y[3], label=metodo, s=15)

plt.xlabel("t")
plt.ylabel("P")
plt.title("Producto calculado con distintos solvers")
plt.legend()
plt.show()


# %%
for metodo in metodos:
    sol_m = soluciones[metodo]
    plt.scatter(sol_m.t, sol_m.y[1], label=metodo, s=15)

plt.xlabel("t")
plt.ylabel("P")
plt.title("Enzima calculado con distintos solvers")
plt.legend()
plt.show()


# %% [markdown]
# # Modelo reducido: Michaelis-Menten
#
# El modelo completo tiene cuatro variables:
#
# $$
# [S], [E], [ES], [P].
# $$
#
# Pero si el complejo $ES$ llega rapido a un estado casi estacionario,
# podemos aproximar:
#
# $$
# \frac{d[ES]}{dt} \approx 0.
# $$
#
# Entonces se obtiene la ley de Michaelis-Menten:
#
# $$
# v([S]) = \frac{V_{max}[S]}{K_m + [S]}
# $$
#
# donde:
#
# $$
# K_m = \frac{k_{-1}+k_{cat}}{k_1}
# $$
#
# y
#
# $$
# V_{max} = k_{cat} [E_{tot}].
# $$


# %%
Km = (km1 + kcat) / k1
Etot = E0 + ES0
Vmax = kcat * Etot

def michaelis_menten(t, y, Vmax, Km):
    S, P = y

    v = Vmax * S / (Km + S)

    dSdt = -v
    dPdt = v

    return np.array([dSdt, dPdt])


# %%
y0_red = np.array([S0, P0])

t_span_largo = (0.0, 300.0)
t_eval_largo = np.linspace(t_span_largo[0], t_span_largo[1], 1000)

sol_full = integrate.solve_ivp(
    enzima_completo,
    t_span_largo,
    y0,
    args=(k1, km1, kcat),
    method="Radau",
    t_eval=t_eval_largo,
    rtol=1e-8,
    atol=1e-12
)

sol_red = integrate.solve_ivp(
    michaelis_menten,
    t_span_largo,
    y0_red,
    args=(Vmax, Km),
    method="RK45",
    t_eval=t_eval_largo,
    rtol=1e-8,
    atol=1e-12
)

print("modelo completo:", sol_full.message)
print("success: ", sol_full.success)
print("status: ", sol_full.status)
print("nfev: ", sol_full.nfev)
print("njev: ", sol_full.njev)
print("modelo reducido:", sol_red.message)
print("success: ", sol_red.success)
print("status: ", sol_red.status)
print("nfev: ", sol_red.nfev)
print("njev: ", sol_red.njev)

# %%
S_full = sol_full.y[0]
P_full = sol_full.y[3]

S_red = sol_red.y[0]
P_red = sol_red.y[1]

plt.plot(sol_full.t, S_full, label="S completo")
plt.plot(sol_red.t, S_red, "--", label="S Michaelis-Menten")
plt.xlabel("t")
plt.ylabel("S")
plt.title("Comparacion del sustrato")
plt.legend()
plt.show()


# %%
plt.plot(sol_full.t, P_full, label="P completo")
plt.plot(sol_red.t, P_red, "--", label="P Michaelis-Menten")
plt.xlabel("t")
plt.ylabel("P")
plt.title("Comparacion del producto")
plt.legend()
plt.show()


# %%
error_S = np.abs(S_full - S_red) / S0
error_P = np.abs(P_full - P_red) / S0

plt.plot(sol_full.t, error_S, label="error en S")
plt.plot(sol_full.t, error_P, label="error en P")
plt.xlabel("t")
plt.ylabel("error relativo")
plt.title("Error del modelo reducido")
plt.legend()
plt.show()


# %% [markdown]
# ## Cuando falla el modelo reducido?
#
# El modelo de Michaelis-Menten funciona mejor cuando la enzima total es pequena
# comparada con la escala del sustrato.
#
# Ahora cambiamos $[E_0]$ y comparamos el modelo completo contra el reducido.


# %%
E0_values = [1e-7, 1e-6, 1e-5, 5e-5]
errores_maximos = []

for E0_test in E0_values:
    y0_test = np.array([S0, E0_test, 0.0, 0.0])

    Etot_test = E0_test
    Vmax_test = kcat * Etot_test

    sol_full_test = integrate.solve_ivp(
        enzima_completo,
        t_span_largo,
        y0_test,
        args=(k1, km1, kcat),
        method="Radau",
        t_eval=t_eval_largo,
        rtol=1e-8,
        atol=1e-12
    )

    sol_red_test = integrate.solve_ivp(
        michaelis_menten,
        t_span_largo,
        np.array([S0, 0.0]),
        args=(Vmax_test, Km),
        method="RK45",
        t_eval=t_eval_largo,
        rtol=1e-8,
        atol=1e-12
    )

    S_full_test = sol_full_test.y[0]
    S_red_test = sol_red_test.y[0]

    error = np.max(np.abs(S_full_test - S_red_test) / S0)
    errores_maximos.append(error)

    print("E0:", E0_test)
    print("E0/S0:", E0_test / S0)
    print("error maximo:", error)
    print()


# %%
plt.plot(np.array(E0_values) / S0, errores_maximos, "o-")
plt.xlabel(r"$E_0/S_0$")
plt.ylabel("error maximo relativo")
plt.title("Validez del modelo reducido")
plt.show()


# %% [markdown]
# # TAREA
#
# 1. Derivar el modelo reducido de Michaelis-Menten usando:
#
# $$
# \frac{d[ES]}{dt} \approx 0.
# $$
#
# 1. Verificar que:
#
# $$
# K_m = \frac{k_{-1}+k_{cat}}{k_1}
# $$
#
# y
#
# $$
# V_{max}=k_{cat}[E_{tot}].
# $$
#
# 1. Cambiar la razon $[E_0]/[S_0]$ y estudiar cuando el modelo reducido
# deja de ser una buena aproximacion.
#
# 1. Cambiar $k_1$, $k_{-1}$ y $k_{cat}$ para producir sistemas mas o menos
# rigidos.
#
# 1. Comparar `RK45`, `Radau`, `BDF` y `LSODA` usando:
#
# - `nfev`
# - `njev`
# - error en las cantidades conservadas
# - tiempo de ejecucion
# %%
