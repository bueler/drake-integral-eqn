from firedrake import *

def formKmat(V, K=None):
    '''Build PETSc Mat for integral operator K by columns
    using the MatType set at runtime.'''
    x = SpatialCoordinate(V.mesh())[0]
    v = TestFunction(V)
    from firedrake.petsc import PETSc
    Kmat = PETSc.Mat()
    Kmat.create(PETSc.COMM_WORLD)
    Kmat.setSizes((V.dim(),V.dim()))
    Kmat.setFromOptions()
    Kmat.setUp()  # or fix type and do preallocation?
    phij = Function(V)
    Y = Constant(0.0)
    N = V.dim() - 1
    hx = 1.0 / N
    for j, yj in enumerate(V.mesh().coordinates.dat.data):
        phij.dat.data[:] = 0.0
        phij.dat.data[j] = 1.0
        # note  c * K(x,Y)  approximates  int_0^1 K(x,y) phi_j(y) dy
        if (j == 0) or (j == N):  c = 0.5 * hx
        else:                     c = hx
        Y.assign(yj)
        Kmat.setValues(range(V.dim()),[j,],
                       assemble(c * K(x,Y) * v * dx).dat.data)
    Kmat.assemblyBegin()
    Kmat.assemblyEnd()
    return Kmat

def solve(N, K=None, f=None):
    '''Solve the Fredholm second-kind integral equation
       u(x) + int_0^1 K(x,y) u(y) dy = f(x)
    for u(x) defined on [0,1], using P1 elements.
    The matrix A and right side b are created by Firedrake
    assemble and by formKmat().  The linear system  A u = b
    is solved using PETSc KSP.  The type of matrix A is
    determined at runtime.'''
    mesh = UnitIntervalMesh(N)
    x = SpatialCoordinate(mesh)[0]
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    b = assemble(f(x) * v * dx)
    Mass = assemble(u * v * dx)
    Amat = Mass.M.handle + formKmat(V, K=K)
    from firedrake.petsc import PETSc
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A=Amat, P=Amat)
    ksp.setFromOptions()
    u = Function(V)
    with u.dat.vec as vu:
        with b.dat.vec_ro as vb:
            ksp.solve(vb, vu)
    return V, x, u

if __name__ == '__main__':
    def K(x,y):
        return - 0.5 * (x + 1) * exp(-x * y)

    def f(x):
        return exp(-x) - 0.5 + 0.5 * exp(-(x+1))

    def uexact(x):
        return exp(-x)

    import time
    import numpy as np
    REFINES = 12
    for N in 2**(3 + np.arange(REFINES)):
        tic = time.time()
        V, x, u = solve(N=N, K=K, f=f)
        toc = time.time()
        uex = Function(V).interpolate(uexact(x))
        e = errornorm(uex, u, 'L2')
        print('  N = %5d: error = %9.3e [%6.1f s]' % (N, e, toc-tic))
