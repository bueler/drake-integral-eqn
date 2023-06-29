from firedrake import *
import numpy as np 

# We consider the integral equation
#
# u(x) - 0.5\int_{0}^{1}(x+1)\exp{-xy}u(y)dy = f(x)   in [0, 1],
#
# where f(x) = \exp{-x} - 0.5 + 0.5\exp{-(x+1)}.  Here the kernel
# is
#    K(x,y) = -0.5 (x+1) exp(-xy)
# and the equation is
#    u(x) + int_0^1 K(x,y) u(y) dy = f(x)
# The problem has an exact solution u(x) = exp(-x).

def fredholm(N):
    mesh = UnitIntervalMesh(N)
    x = SpatialCoordinate(mesh)[0]
    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    f = exp(-x) - 0.5 + 0.5 * exp(-(x+1))
    b = assemble(f * v * dx).dat.data
    # build dense A by columns
    A = np.zeros((N+1,N+1))
    phij = Function(V)
    hx = 1.0 / N
    Y = Constant(0.0)
    Kj = -0.5 * (x + 1) * exp(- x * Y)
    for j, yj in enumerate(mesh.coordinates.dat.data):
        phij.dat.data[:] = 0.0
        phij.dat.data[j] = 1.0
        Y.assign(yj)
        if (j == 0) or (j == N):
            c = 0.5 * hx
        else:
            c = hx
        # c * Kj approximates  int_0^1 K(x,y) phi_j(y) dy:
        A[:, j] = assemble(phij * v * dx +
                           c * Kj * v * dx).dat.data
    # solve A u = b
    u = Function(V)
    u.dat.data[:] = np.linalg.solve(A, b)
    u_exact = Function(V).interpolate(exp(-x))
    return errornorm(u_exact, u, 'L2')

if __name__ == '__main__':
    for N in [8,16,32,64,128,256]:
        e = fredholm(N=N)
        print('N=%3d: hx = %.6f,  error=%.3g' % (N, 1.0/N, e))
