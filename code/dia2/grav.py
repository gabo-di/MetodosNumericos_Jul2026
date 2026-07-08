# %% [markdown]
# # Ecuaciones diferenciales ordinarias




# %% [markdown]
# ## Objetivos
# - Resolver ODEs con Python
#
# - Entender problemas de valor inicial
#
# - Comparar integradores numericos
#
# - Ver la diferencia entre integradores generales e integradores simplecticos




# %% [markdown]
# # El problema de los dos cuerpos
#
# Usaremos coordenadas relativas.
#
# $$
# \mathbf r = \mathbf r_2 - \mathbf r_1
# $$
#
# entonces la ecuacion de movimiento es
#
# $$
# \frac{d^2\mathbf r}{dt^2}
# =
# -\frac{G(m_1+m_2)}{r^3}\mathbf r.
# $$
#
# Definimos
#
# $$
# \mu = G(m_1+m_2).
# $$




# %% [markdown]
# ## Sistema de primer orden
#
# Integramos en coordenadas cartesianas:
#
# $$
# (\mathbf r, \, \mathbf v) = (x, y, v_x, v_y).
# $$
#
# Entonces
#
# $$
# \frac{dx}{dt}=v_x,
# \qquad
# \frac{dy}{dt}=v_y,
# $$
#
# $$
# \frac{dv_x}{dt}
# =
# -\frac{\mu x}{(x^2+y^2)^{3/2}},
# \qquad
# \frac{dv_y}{dt}
# =
# -\frac{\mu y}{(x^2+y^2)^{3/2}}.
# $$
#
# Las cantidades conservadas son:
# 
# 1. La energia especifica:
#
# $$
# E = \frac{1}{2}(v_x^2+v_y^2)-\frac{\mu}{r}.
# $$
#
# 1. El momento angular especifico:
#
# $$
# L_z = x v_y - y v_x.
# $$




# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# parametros global/constante 
mu = 1.0

def kepler_ode(t, y, mu):
    x, yy, vx, vy = y
    r = np.sqrt(x**2 + yy**2)

    dxdt = vx
    dydt = vy
    dvxdt = ???
    dvydt = ???

    return np.array([dxdt, dydt, dvxdt, dvydt])

def energia(y, mu):
    x = y[0]
    yy = y[1]
    vx = y[2]
    vy = y[3]

    r = np.sqrt(x**2 + yy**2)
    return ???


def momento_angular(y):
    x = y[0]
    yy = y[1]
    vx = y[2]
    vy = y[3]

    return ???




# %% [markdown]
# ## Condiciones iniciales desde elementos orbitales
#
# Para una orbita eliptica usamos:
#
# $$
# a = \text{semi-eje mayor},
# \qquad
# e = \text{excentricidad}.
# $$
#
# Empezamos en el periastro:
#
# $$
# r_p = a(1-e).
# $$
#
# La velocidad en el periastro es:
#
# $$
# v_p =
# \sqrt{
# \mu
# \frac{1+e}{a(1-e)}
# }.
# $$
#
# El periodo orbital es:
#
# $$
# T = 2\pi \sqrt{\frac{a^3}{\mu}}.
# $$




# %%
def condicion_inicial(a, e, mu):
    rp = a * (1 - e)
    vp = np.sqrt(mu * (1 + e) / (a * (1 - e)))
    return np.array([rp, 0.0, 0.0, vp])

def periodo(a, mu):
    return 2 * np.pi * np.sqrt(a**3 / mu)




# %% [markdown]
# ## Una orbita con `solve_ivp`
#
# Primero usamos un integrador general de SciPy.




# %%
a = 1.0
e = 0.9
n_orbitas = 1000
n_points = 10000

y0 = condicion_inicial(a=a, e=e, mu=mu)
T = periodo(a=a, mu=mu)

t_span = (0.0, n_orbitas * T)
t_eval = np.linspace(t_span[0], t_span[1], n_points)

sol = integrate.solve_ivp(
    kepler_ode,
    t_span,
    y0,
    args=(mu,),
    method="RK45",
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-8
)

print(sol.message)
print("success: ", sol.success)
print("status: ", sol.status)
print("nfev: ", sol.nfev)

plt.scatter(sol.y[0], sol.y[1], s=1)
plt.scatter([0], [0], s=8, color="red")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Orbita con RK45")
plt.show()

E = energia(sol.y, mu=mu)
L = momento_angular(sol.y)

print("E inicial: ", E[0])
print("E final: ", E[-1])
print("L inicial: ", L[0])
print("L final: ", L[-1])




# %%
plt.plot(sol.t / T, (E - E[0]) / abs(E[0]))
plt.xlabel("t / T")
plt.ylabel(r"$(E - E_0)/|E_0|$")
plt.title("Error relativo de energia")
plt.show()




# %%
plt.plot(sol.t / T, (L - L[0]) / abs(L[0]))
plt.xlabel("t / T")
plt.ylabel(r"$(L - L_0)/|L_0|$")
plt.title("Error relativo de momento angular")
plt.show()




# %% [markdown]
# # Integradores simplecticos
#
# Los integradores simplécticos conservan la estructura simpléctica del espacio de fases.
# En una dimensión esto puede visualizarse como conservación de área en el plano
# \((q,p)\). En dimensiones mayores, la propiedad correcta es conservar la forma
# simpléctica, no solamente el volumen.
#
# REBOUND es una libreria especializada para problemas gravitacionales
# de N cuerpos, consta con integradores simplecticos.
# nota usaremos los siguientes integradores:
# - whfast (simplectico)    
# - ias15 (no simplectico pero de gran precision para este tipo de simulaciones)
# - leapfrog (simplectico)



# %%
import rebound

def norma_vec3(v):
    return np.sqrt(v.x**2 + v.y**2 + v.z**2)

def crear_simulacion_dos_cuerpos(integrator, a, e, dt, mu):
    sim = rebound.Simulation()
    sim.G = mu

    sim.add(m=1.0)
    sim.add(m=1e-6, a=a, e=e)

    sim.move_to_com()

    sim.integrator = integrator
    sim.dt = dt

    return sim

def integrar_rebound(sim, tiempos, exact_finish_time=0):
    posiciones = np.zeros((len(tiempos), sim.N, 3))
    energias = np.zeros(len(tiempos))
    momentos = np.zeros(len(tiempos))
    tiempos_reales = np.zeros(len(tiempos))

    for i, t in enumerate(tiempos):
        sim.integrate(t, exact_finish_time=exact_finish_time)

        tiempos_reales[i] = sim.t

        for j, p in enumerate(sim.particles):
            posiciones[i, j, 0] = p.x
            posiciones[i, j, 1] = p.y
            posiciones[i, j, 2] = p.z

        energias[i] = sim.energy()
        momentos[i] = norma_vec3(sim.angular_momentum())

    return tiempos_reales, posiciones, energias, momentos




# %%
T = periodo(a=a, mu=mu)

tiempos = np.linspace(0.0, n_orbitas*T, n_points)

integradores = ["whfast", "ias15", "leapfrog"]
resultados = {}

for integrador in integradores:
    sim = crear_simulacion_dos_cuerpos(
        integrator=integrador,
        a=a,
        e=e,
        dt=T/1000,
        mu=mu
    )

    tiempos_reales, pos, E, L = integrar_rebound(sim, tiempos)
    resultados[integrador] = {
        "tiempos_reales": tiempos_reales,
        "pos": pos,
        "E": E,
        "L": L
    }

    print(integrador, "terminado")

for integrador in integradores:
    pos = resultados[integrador]["pos"]

    x_rel = pos[:, 1, 0] - pos[:, 0, 0]
    y_rel = pos[:, 1, 1] - pos[:, 0, 1]

    plt.scatter(x_rel, y_rel, label=integrador, s=1)

plt.scatter([0], [0], s=8, color="red")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Orbitas con REBOUND")
plt.show()




# %%
for integrador in integradores:
    tiempos_reales = resultados[integrador]["tiempos_reales"]
    E = resultados[integrador]["E"]
    error_E = np.abs((E - E[0]) / E[0])

    plt.plot(tiempos_reales / T, error_E + 1e-30, label=integrador)

plt.xlabel("t / T")
plt.ylabel(r"$(E - E_0)/|E_0|$")
plt.title("Error relativo de energia")
plt.legend()
plt.show()




# %%
for integrador in integradores:
    tiempos_reales = resultados[integrador]["tiempos_reales"]
    L = resultados[integrador]["L"]
    error_L = np.abs((L - L[0]) / L[0])

    plt.plot(tiempos_reales / T, error_L + 1e-30, label=integrador)

plt.xlabel("t / T")
plt.ylabel(r"$(L - L_0)/|L_0|$")
plt.legend()
plt.title("Error relativo de momento angular")
plt.show()



# %% [markdown]
# ## Energia vs momento angular
#
# Un integrador simplectico no necesariamente conserva exactamente la energia
# original del sistema. En general, conserva exactamente la geometria Hamiltoniana
# $ dq \wedge dp = dQ \wedge dP $,
# y se puede interpretar como si conservara un Hamiltoniano modificado.
#
# Por eso el error de energia puede oscilar.
#
# El momento angular es distinto. En el problema de Kepler la fuerza es central:
#
# $$
# \mathbf F(\mathbf r) = -\frac{\mu}{r^3}\mathbf r.
# $$
#
# Entonces
#
# $$
# \mathbf r \times \mathbf F = 0.
# $$
#
# El metodo leapfrog conserva muy bien el momento angular porque sus pasos de
# "kick" y "drift" respetan esta simetria rotacional. Por eso, aunque el error
# de energia pueda ser mayor que en RK45, el momento angular puede ser mucho
# mejor conservado.



# %% [markdown]
# # TAREA
#
# 1. como cambia el error de energia y el error de momento angular, si
# se usan otros valores de excentricidad?
#
# 1. como cambia el error de energia y el error de momento angular, si
# se usan otros pasos de tiempo `dt`?
#
# 1. si el problema es de tres cuerpos, rk45 es un buen integrador?
# (probar ejemplos con close encounters)
#
# 1. para el problema de tres cuerpos con close encounters prueba
# otros integradores de la libreria REBOUND.