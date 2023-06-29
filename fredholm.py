from firedrake import *

def fredholm_numpy(N, K=None, f=None):
    '''Solve the Fredholm second-kind integral equation
       u(x) + int_0^1 K(x,y) u(y) dy = f(x)
    for u(x) defined on [0,1], using P1 elements.
    The linear system  A u = b  is solved using the numpy
    direct solver.'''
    mesh = UnitIntervalMesh(N)
    x = SpatialCoordinate(mesh)[0]
    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    b = assemble(f(x) * v * dx).dat.data
    # build dense A by columns
    import numpy as np
    A = np.zeros((N+1,N+1))
    phij = Function(V)
    hx = 1.0 / N
    Y = Constant(0.0)
    for j, yj in enumerate(mesh.coordinates.dat.data):
        phij.dat.data[:] = 0.0
        phij.dat.data[j] = 1.0
        # note  c * K(x,Y) approximates  int_0^1 K(x,y) phi_j(y) dy
        if (j == 0) or (j == N):  c = 0.5 * hx
        else:                     c = hx
        Y.assign(yj)
        A[:, j] = assemble(phij * v * dx +
                           c * K(x,Y) * v * dx).dat.data
    u = Function(V)
    u.dat.data[:] = np.linalg.solve(A, b)
    return V, x, u

def fredholm_petsc(N, K=None, f=None):
    '''Solve the Fredholm second-kind integral equation
       u(x) + int_0^1 K(x,y) u(y) dy = f(x)
    for u(x) defined on [0,1], using P1 elements.
    The linear system  A u = b  is solved using PETSc
    KSP.  The type of matrix A is determined at runtime.'''
    from firedrake.petsc import PETSc
    mesh = UnitIntervalMesh(N)
    x = SpatialCoordinate(mesh)[0]
    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    b = PETSc.Vec()
    b.create(PETSc.COMM_WORLD)
    b.setSizes(N+1)
    b.setFromOptions()
    b.setValues(range(N+1), assemble(f(x) * v * dx).dat.data)
    b.assemblyBegin()
    b.assemblyEnd()
    # build A by columns using MatType set at runtime
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes((N+1, N+1))
    A.setFromOptions()
    A.setUp()  # or fix type and do preallocation?
    phij = Function(V)
    hx = 1.0 / N
    Y = Constant(0.0)
    for j, yj in enumerate(mesh.coordinates.dat.data):
        phij.dat.data[:] = 0.0
        phij.dat.data[j] = 1.0
        # note  c * K(x,Y) approximates  int_0^1 K(x,y) phi_j(y) dy
        if (j == 0) or (j == N):  c = 0.5 * hx
        else:                     c = hx
        Y.assign(yj)
        A.setValues(range(N+1),[j,],
                    assemble(phij * v * dx + c * K(x,Y) * v * dx).dat.data)
    A.assemblyBegin()
    A.assemblyEnd()
    # solve using PETSc KSP
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setOperators(A=A, P=A)
    ksp.setFromOptions()
    u = Function(V)
    with u.dat.vec as vu:
        ksp.solve(b, vu)
    return V, x, u

if __name__ == '__main__':
    def K(x,y):
        return - 0.5 * (x + 1) * exp(-x * y)

    def f(x):
        return exp(-x) - 0.5 + 0.5 * exp(-(x+1))

    def uexact(x):
        return exp(-x)

    print('**** numpy direct solve ****')
    for N in [8,16,32,64,128,256,512,1024]:
        V, x, u = fredholm_numpy(N=N, K=K, f=f)
        uex = Function(V).interpolate(uexact(x))
        e = errornorm(uex, u, 'L2')
        print('  N=%3d: error=%.3g' % (N, e))

    print('**** petsc ksp solve ****')
    for N in [8,16,32,64,128,256,512,1024]:
        V, x, u = fredholm_petsc(N=N, K=K, f=f)
        uex = Function(V).interpolate(uexact(x))
        e = errornorm(uex, u, 'L2')
        print('  N=%3d: error=%.3g' % (N, e))
