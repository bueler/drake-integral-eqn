from firedrake import *

def solve(N, K=None, f=None):
    '''Solve the Fredholm second-kind integral equation
       u(x) + int_0^1 K(x,y) u(y) dy = f(x)
    for u(x) defined on [0,1], using P1 elements.  numpy
    arrays are used for (dense) A and b in the linear system
    A u = b.  It is solved using the numpy direct solver.'''
    mesh = UnitIntervalMesh(N)
    x = SpatialCoordinate(mesh)[0]
    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    b = assemble(f(x) * v * dx).dat.data
    import numpy as np
    A = np.zeros((N+1,N+1))
    phij = Function(V)
    hx = 1.0 / N
    Y = Constant(0.0)
    for j, xj in enumerate(mesh.coordinates.dat.data):
        phij.dat.data[:] = 0.0
        phij.dat.data[j] = 1.0
        # note  c * K(x,Y) approximates  int_0^1 K(x,y) phi_j(y) dy
        if (j == 0) or (j == N):  c = 0.5 * hx
        else:                     c = hx
        Y.assign(xj)
        A[:, j] = assemble(phij * v * dx +
                           c * K(x,Y) * v * dx).dat.data
    u = Function(V)
    u.dat.data[:] = np.linalg.solve(A, b)
    return V, x, u

if __name__ == '__main__':
    def K(x,y):
        return - 0.5 * (x + 1) * exp(-x * y)

    def f(x):
        return exp(-x) - 0.5 + 0.5 * exp(-(x+1))

    def uexact(x):
        return exp(-x)

    import sys
    import time
    import numpy as np
    N = int(sys.argv[1])
    tic = time.time()
    V, x, u = solve(N=N, K=K, f=f)
    toc = time.time()
    uex = Function(V).interpolate(uexact(x))
    e = errornorm(uex, u, 'L2')
    print('N = %d: error = %9.3e [%6.1f s]' % (N, e, toc-tic))
