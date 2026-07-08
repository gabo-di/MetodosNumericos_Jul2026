# %% [markdown]
# # Optimizacion




# %% [markdown]
# ## Objetivos
# - Comprender los conceptos basicos de optimizacion
# 
# - Aplicar los conceptos basicos de optimizacion a problemas de fisica



# %% [markdown]
# # El potencial de Lennard-Jones
#
# El potencial de Lennard-Jones es uno de los modelos más simples y usados para describir la interacción entre átomos o moléculas neutras. Aunque es una aproximación, captura dos efectos físicos importantes:
#
# 1. **Atracción a distancias intermedias:**
#    Incluso dos átomos neutros pueden atraerse debido a fluctuaciones instantáneas de sus nubes electrónicas. Estas fluctuaciones producen dipolos inducidos, dando lugar a fuerzas de dispersión de London. Esta contribución atractiva decae aproximadamente como
#
# $$
# -\frac{1}{r^6}.
# $$
#
# 2. **Repulsión a distancias cortas:**
#    Cuando dos átomos se acercan demasiado, sus nubes electrónicas se solapan. Esto produce una repulsión muy fuerte asociada al principio de exclusión de Pauli. En muchos modelos, esta repulsión se aproxima mediante un término muy empinado,
#
# $$
# \frac{1}{r^{12}}.
# $$
#
# Por eso el potencial completo se escribe como
#
# $$
# V(r)=4\epsilon \left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right].
# $$
#
# Aquí, ($\epsilon$) controla la profundidad del pozo de energía, y ($\sigma$) es la distancia donde el potencial cambia de signo. El mínimo ocurre en
#
# $$
# r_{\min}=2^{1/6}\sigma,
# $$
#
# y en ese punto
#
# $$
# V(r_{\min})=-\epsilon.
# $$
#
# Este modelo es importante porque aparece como bloque básico en simulaciones moleculares. En sistemas biológicos reales, como proteínas, membranas o complejos proteína-ligando, el potencial total no es solamente Lennard-Jones. También se incluyen enlaces, ángulos, torsiones y fuerzas electrostáticas. Sin embargo, el término de Lennard-Jones sigue siendo una parte esencial de los modelos de campo de fuerza porque representa las interacciones de van der Waals entre átomos no enlazados.




# %% [markdown]
# ## Lennard-Jones potencial - 1D
#
# $$
# V(r) = 4 \epsilon \left( \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right)
# $$




# %%
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np

# parametros globales y constantes
# de otra manera se usaria como variables en las funciones
epsilon = 1.0
sigma = 1.0

# Lennard-Jones potencial - 1D
def V_1D(r):
    return 4 * epsilon * (np.power(sigma / r, 12) - np.power(sigma / r, 6))

# grafica del potencial
r = np.linspace(0.9, 3.0, 100)
v = V_1D(r)
plt.plot(r, v)
plt.xlabel('r')
plt.ylabel('V(r)')
plt.show()




# %% [markdown]
# ## Minimizacion del potencial
#
# usamos `scipy.optimize.minimize` para minimizar el potencial.
#
# NOTAS:
#
# 1. se necesita un punto de inicio para la minimizacion `x0=1`
#
# 1. se usa el metodo de BFGS para la minimizacion `method='BFGS'`
# otras opciones son `'Nelder-Mead'`, `'L-BFGS-B'`
# 
# 1. no proporcionamos la derivada, opt usa diferencias finitas, para
# mayor precision se puede proporcionar la derivada exacta




# %%
res_1 = opt.minimize(V_1D, x0=1.0, method='BFGS')
print(res_1.message)
print("success: ", res_1.success)
print("status: ", res_1.status)
print("nit: ", res_1.nit)
print("nfev: ", res_1.nfev)
print("njev: ", res_1.njev)

plt.plot(r, v)
plt.plot(res_1.x, V_1D(res_1.x), 'ro')
plt.xlabel('r')
plt.ylabel('V(r)')
plt.show()




# %% [markdown]
# ## Lennard-Jones potencial - 3D x N particulas
#
# $$
# V = 4 \epsilon \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \left( \left( \frac{\sigma}{||r_i - r_j||} \right)^{12} - \left( \frac{\sigma}{||r_i - r_j||} \right)^{6} \right)
# $$
#




# %%
def V(x):
    positions = x.reshape(-1, 3)
    N = positions.shape[0]
    V = 0.0
    for i in range(N-1):
        for j in range(i+1, N):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            V += 4 * epsilon * (np.power(sigma / r_ij, 12) - np.power(sigma / r_ij, 6))
    return V




# %% [markdown]
# ## 3D x 3 particulas




# %%
positions = np.array([[2.0, 3.0, 4.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).flatten()
res_3 = opt.minimize(V, x0=positions, method='BFGS')
print(res_3.message)
print("success: ", res_3.success)
print("status: ", res_3.status)
print("nit: ", res_3.nit)
print("nfev: ", res_3.nfev)
print("njev: ", res_3.njev)

print(res_3.x.reshape(-1, 3))
print(V(res_3.x))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(res_3.x.reshape(-1, 3)[:, 0], res_3.x.reshape(-1, 3)[:, 1], res_3.x.reshape(-1, 3)[:, 2])
plt.show()




# %%
# posiciones relativas entre partículas
r_12 = (res_3.x.reshape(-1, 3)[0] - res_3.x.reshape(-1, 3)[1])
r_13 = (res_3.x.reshape(-1, 3)[0] - res_3.x.reshape(-1, 3)[2])
r_23 = (res_3.x.reshape(-1, 3)[1] - res_3.x.reshape(-1, 3)[2])

# distancias entre partículas
print("distancia 12: ", np.linalg.norm(r_12))
print("distancia 13: ", np.linalg.norm(r_13))
print("distancia 23: ", np.linalg.norm(r_23))
print(np.linalg.norm(r_23))

# coseno del angulo entre partículas
print("coseno del angulo 12-13: ", np.dot(r_12, r_13) / (np.linalg.norm(r_12) * np.linalg.norm(r_13)))
print("coseno del angulo 13-23: ", np.dot(r_13, r_23) / (np.linalg.norm(r_13) * np.linalg.norm(r_23)))
print("coseno del angulo 23-12: ", np.dot(r_23, -r_12) / (np.linalg.norm(r_23) * np.linalg.norm(r_12)))




# %% [markdown]
# ## 3D x 4 particulas




# %%
positions = np.array([[2.0, 3.0, 4.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).flatten()
print(V(positions))

res_4 = opt.minimize(V, x0=positions, method='BFGS')
print(res_4.message)
print("success: ", res_4.success)
print("status: ", res_4.status)
print("nit: ", res_4.nit)
print("nfev: ", res_4.nfev)
print("njev: ", res_4.njev)

print(res_4.x.reshape(-1, 3))
print(V(res_4.x))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(res_4.x.reshape(-1, 3)[:, 0], res_4.x.reshape(-1, 3)[:, 1], res_4.x.reshape(-1, 3)[:, 2])
plt.show()




# %%
# posiciones relativas entre partículas
r_12 = (res_4.x.reshape(-1, 3)[0] - res_4.x.reshape(-1, 3)[1])
r_13 = (res_4.x.reshape(-1, 3)[0] - res_4.x.reshape(-1, 3)[2])
r_14 = (res_4.x.reshape(-1, 3)[0] - res_4.x.reshape(-1, 3)[3])
r_23 = (res_4.x.reshape(-1, 3)[1] - res_4.x.reshape(-1, 3)[2])
r_24 = (res_4.x.reshape(-1, 3)[1] - res_4.x.reshape(-1, 3)[3])
r_34 = (res_4.x.reshape(-1, 3)[2] - res_4.x.reshape(-1, 3)[3])

# distancias entre partículas
print("distancia 12: ", np.linalg.norm(r_12))
print("distancia 13: ", np.linalg.norm(r_13))
print("distancia 14: ", np.linalg.norm(r_14))
print("distancia 23: ", np.linalg.norm(r_23))
print("distancia 24: ", np.linalg.norm(r_24))
print("distancia 34: ", np.linalg.norm(r_34))

# coseno del angulo entre partículas
print("coseno del angulo 12-13: ", np.dot(r_12, r_13) / (np.linalg.norm(r_12) * np.linalg.norm(r_13)))
print("coseno del angulo 13-14: ", np.dot(r_13, r_14) / (np.linalg.norm(r_13) * np.linalg.norm(r_14)))
print("coseno del angulo 14-12: ", np.dot(r_14, r_12) / (np.linalg.norm(r_14) * np.linalg.norm(r_12)))
print("coseno del angulo 23-24: ", np.dot(r_23, r_24) / (np.linalg.norm(r_23) * np.linalg.norm(r_24)))
print("coseno del angulo 24-23: ", np.dot(r_24, r_23) / (np.linalg.norm(r_24) * np.linalg.norm(r_23)))
print("coseno del angulo 34-32: ", np.dot(r_34, -r_23) / (np.linalg.norm(r_34) * np.linalg.norm(r_23)))
print("coseno del angulo 32-34: ", np.dot(r_23, -r_34) / (np.linalg.norm(r_23) * np.linalg.norm(r_34)))




# %% [markdown]
# ## 3D x 7 particulas




# %%
np.random.seed(42)
positions = np.random.rand(7, 3).flatten()
print(V(positions))

res_7 = opt.minimize(V, x0=positions, method='BFGS')
print(res_7.message)
print("success: ", res_7.success)
print("status: ", res_7.status)
print("nit: ", res_7.nit)
print("nfev: ", res_7.nfev)
print("njev: ", res_7.njev)

print(V(res_7.x))




# %% [markdown]
# `BFGS` no es suficiente
# 
# alternativa: `L-BFGS-B`




# %%
res_7 = opt.minimize(V, x0=positions, method='L-BFGS-B')
print(res_7.message)
print("success: ", res_7.success)
print("status: ", res_7.status)
print("nit: ", res_7.nit)
print("nfev: ", res_7.nfev)
print("njev: ", res_7.njev)

print(V(res_7.x)) # -15.53306




# %% [markdown]
# El resultade es mejor, pero no es correcto,
# en la literatura el valor esperado es $LJ_{7}=-16.505384$




# %% [markdown]
# # TAREA




# %% [markdown]
# ## Optimizacion Global
#
# 1. Estudia la diferencia entre minimos locales y globales




# %% [markdown]
# `scipy.optimize.minimize` es un optimizador local,
# para optimizacion global se puede usar `scipy.optimize.basinhopping`
# para obtener mas información sobre `scipy.optimize.basinhopping`
# se puede usar `scipy.optimize.basinhopping?`
#
# Docstring:
#
# Find the global minimum of a function using the basin-hopping algorithm.
#
# Basin-hopping is a two-phase method that combines a global stepping
# algorithm with local minimization at each step. Designed to mimic
# the natural process of energy minimization of clusters of atoms, it works
# well for similar problems with "funnel-like, but rugged" energy landscapes
# [5]_.
#
# As the step-taking, step acceptance, and minimization methods are all
# customizable, this function can also be used to implement other two-phase
# methods.
#
# This global minimization method has been shown to be extremely efficient
# for a wide variety of problems in physics and chemistry. It is
# particularly useful when the function has many minima separated by large
# barriers. See the `Cambridge Cluster Database
# <https://www-wales.ch.cam.ac.uk/CCD.html>`_ for databases of molecular
# systems that have been optimized primarily using basin-hopping. This
# database includes minimization problems exceeding 300 degrees of freedom.
#
# Choosing `stepsize`:  This is a crucial parameter in `basinhopping` and
# depends on the problem being solved. The step is chosen uniformly in the
# region from x0-stepsize to x0+stepsize, in each dimension. Ideally, it
# should be comparable to the typical separation (in argument values) between
# local minima of the function being optimized. `basinhopping` will, by
# default, adjust `stepsize` to find an optimal value, but this may take
# many iterations. You will get quicker results if you set a sensible
# initial value for ``stepsize``.




# %%
res_7 = opt.basinhopping(V, x0=positions)
print(res_7.message)
print("success: ", res_7.success)
print("nit: ", res_7.nit)
print("nfev: ", res_7.nfev)
print("njev: ", res_7.njev)
print(V(res_7.x))



# %% [markdown]
# `basinhopping` es un optimizador global, pero no es suficiente,
# se necesita mejorar sus kwargs
#
# 1. `method`: metodo de busqueda local
# 1. `options`: opciones de la busqueda local
# 1. `minimizer_kwargs`: kwargs para el metodo de busqueda local
# 1. `niter`: numero de iteraciones
# 1. `stepsize`: tamaño del paso
# 1. `T`: temperatura
#
# Varia los kwargs y observa el resultado, obtiene el valor esperado
# $LJ_{7}=-16.505384$




# %%
# basinhopping con kwargs




# %% [markdown]
# Investiga como usar las derivadas exactas para mejorar el resultado.
#
# 1. programando la derivada exacta
#
# 1. usando JAX para autodiferenciacion




# %%
# derivada exacta programada + basinhopping




# %%
# derivadas con JAX + basinhopping
# %%
