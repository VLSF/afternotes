import jax.numpy as jnp
import numpy as np
import itertools

from jax import random, vmap, jit
from jax.lax import scan
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from jax import random, vmap, jit
from jax.lax import scan

def get_frequencies_and_weights(N, p, l):
    f = 2*jnp.pi*jnp.array([*itertools.product(range(N), repeat=2)])
    w = 1/(1 + (jnp.sum(f, 1)/l)**2)**p
    return f, w

def random_polynomial(x, y, freq, weight, coeff):
    return jnp.real(jnp.sum(jnp.exp(1j*freq[:, 0]*x + 1j*freq[:, 1]*y)*coeff*weight))

@jit
def diffusion_coefficient(x, y, freq, weight, coeff):
    return jnp.exp(random_polynomial(x, y, freq, weight, coeff))

def FD_2D(N_points, F):
    N = M = N_points
    sigma, f = F
    def get_indices_scan(carry, c):
        coords, h_x, h_y, N, M = carry
        row = [c,]*5
        col = []
        val = []
        for i in [+1, -1]:
            x, y = coords[c[0], c[1]]
            val.append(-sigma(x, y + h_y*i/2) / h_y**2)
            col.append(jnp.array([c[0], c[1] + i]))
            val.append(-sigma(x + h_x*i/2, y) / h_x**2)
            col.append(jnp.array([c[0] + i, c[1]]))
        val.append(-sum(val))
        col.append(c)
        return carry, [jnp.array(row), jnp.array(col), jnp.array(val)]

    x = jnp.linspace(0, 1, N+2)[1:-1]
    y = jnp.linspace(0, 1, M+2)[1:-1]
    h_x = x[1] - x[0]
    h_y = y[1] - y[0]
    coords = jnp.stack(jnp.meshgrid(x, y, indexing='ij'), 2)
    ind = jnp.stack(jnp.meshgrid(jnp.arange(N), jnp.arange(M), indexing='ij'), 2).reshape(-1, 2)

    carry = coords, h_x, h_y, N, M
    carry, [rows, cols, vals] = scan(get_indices_scan, carry, ind)

    good_indices = (cols[:, :, 0] != -1) * (cols[:, :, 0] != N) * (cols[:, :, 1] != -1) * (cols[:, :, 1] != M)
    good_indices = good_indices * (rows[:, :, 0] != -1) * (rows[:, :, 0] != N) * (rows[:, :, 1] != -1) * (rows[:, :, 1] != M)

    rows = rows[good_indices]
    cols = cols[good_indices]
    vals = vals[good_indices]

    rows = rows[:, 1] + rows[:, 0] * M
    cols = cols[:, 1] + cols[:, 0] * M
    lex_ind = ind[:, 1] + ind[:, 0] * M

    rows, cols, vals = np.array(rows), np.array(cols), np.array(vals)

    ind = np.stack([rows, cols], 1)
    A = coo_matrix((vals, (rows, cols)), shape=(N*M, N*M)).tocsr()
    b = vmap(f, in_axes=(0, 0))(coords[:, :, 0].reshape(-1,), coords[:, :, 1].reshape(-1,))
    sol = spsolve(A, b)
    sigma_coeff = vmap(sigma, in_axes=(0, 0))(coords[:, :, 0].reshape(-1,), coords[:, :, 1].reshape(-1,)).reshape(N_points, N_points)
    features = jnp.stack([sigma_coeff, jnp.array(b).reshape(N_points, N_points)], 0)
    target = jnp.array(sol).reshape(1, N_points, N_points)
    return features, target, jnp.transpose(coords, (2, 0, 1))

def get_dataset_elliptic_2D(N_samples=1000, key=random.PRNGKey(33), N=40, N_x=64, p=2, l=8, normalize=False):
    f, w = get_frequencies_and_weights(N, p, l)
    keys = random.split(key)
    c1 = random.normal(keys[0], (N_samples, N**2,), dtype=jnp.complex64)
    c2 = random.normal(keys[1], (N**2,), dtype=jnp.complex64)
    F = []
    T = []
    for i in range(N_samples):
        sigma = lambda x, y: diffusion_coefficient(x, y, f, w, c1[i])
        rhs = lambda x, y: random_polynomial(x, y, f, w, c2[i])
        features, target, coordinates = FD_2D(N_x, [sigma, rhs])
        F.append(features)
        T.append(target)
    F = jnp.stack(F, 0)
    T = jnp.stack(T, 0)

    if normalize:
        features_scale = jnp.expand_dims(jnp.max(jnp.max(jnp.abs(F.reshape(F.shape[0], F.shape[1], -1)), axis=2, keepdims=True), axis=0, keepdims=True), 3)
        targets_scale = jnp.expand_dims(jnp.max(jnp.max(jnp.abs(T.reshape(T.shape[0], T.shape[1], -1)), axis=2, keepdims=True), axis=0, keepdims=True), 3)
        F = F / features_scale
        T = T / targets_scale

    return F, T, coordinates

if __name__=="__main__":
    features, targets, coordinates = get_dataset_elliptic_2D()
    data = {
        "features": features,
        "targets": targets,
        "coordinates": coordinates
    }
    jnp.savez("Elliptic_smooth.npz", **data)
