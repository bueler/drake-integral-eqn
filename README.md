# drake-integral-eqn

The [Firedrake](https://www.firedrakeproject.org) finite element library is usually used to discretize the weak forms of partial differential equations (PDEs), but here we want to experiment with integral equations.  This is a bit awkward.

For now we consider only linear integral equations.  The basic strategy is to use Firedrake tools to assemble parts of the discrete linear system.  However, the matrix corresponding to the integral operator is assembled column-by-column in a somewhat "by hand" manner; this is the awkward part.

## example

A well-understood type of linear integral equation is a Fredholm "second kind" equation:
  $$u(x) + \int_0^1 K(x,y) u(y) dy = f(x)$$
We seek $u \in L^2(0,1)$ for given functions $K \in C([0,1]^2)$ and $f\in L^2(0,1)$.

The weak form of the above equation is
  $$\int_0^1 u(x) v(x) dx + \int_0^1 \int_0^1 K(x,y) u(y) v(x) dy dx = \int_0^1 f(x) v(x) dx$$
The first integral is the well-known mass matrix, which has clear Firedrake assembly:

        V = FunctionSpace(mesh, 'CG', 1)  # for example
        u = TrialFunction(V)
        v = TestFunction(V)
        Mass = assemble(u * v * dx)

Likewise, the right-hand side of the weak form is standard:

        x = SpatialCoordinate(mesh)[0]
        b = assemble(f(x) * v * dx)

For the integral operator, however, we assemble a matrix column-by-column by expanding the trial function argument.  That is, we expand $u$ in basis (hat) functions:
  $$u(x) = \sum c_j \phi_j(x)$$
where $\phi_j(x)$ is the hat function at $x_j$.  Inserting this gives a one-form which corresponds to a column of the matrix $K$:
  $$K_{:,j} = \int_0^1 \tilde K_j(x) v(x) dx$$
where
  $$\tilde K_j(x) \approx \int_0^1 K(x,y) \phi_j(y) dy$$
The code we have in mind is this:

        phij = Function(V)
        Y = Constant(0.0)
        for j, yj in enumerate(V.mesh().coordinates.dat.data):
            phij.dat.data[:] = 0.0
            phij.dat.data[j] = 1.0
            Y.assign(yj)
            Kmat.setValues(range(V.dim()),[j,],
                           assemble(c * K(x,Y) * v * dx).dat.data)

Here the constant `c` is the mesh spacing in 1D, and generally it is the volume of a cell around $x_j$; we are using a kind of weighted midpoint rule:
  $$\tilde K_j(x) = h_x K(x,x_j) \approx \int_0^1 K(x,y) \phi_j(y) dy$$
(This formula is modified at the boundary.)

There may well be a better approach.

## solvers

`fredholm.py`: Implements the above approach using Firedrake `assemble()` as much as possible, but generating a PETSc `Mat` using direct petsc4py calls, and then a direct call to `kspsolve()`.

`fredholm_numpy.py`:  In this naive approach we turn everything into [numpy](https://numpy.org/) arrays and call `numpy.linalg.solve()` on it.  This approach works, but it is inflexible with respect to the solver; our choice is limited to the numpy solver.

## demos

*Warning.* These demos should be run with a process/system monitor so the high-resolution cases can be killed.  That is, if you either don't want to wait or you are running out of memory.

Here is the naive numpy approach, which actually has good performance because the matrices are dense.  Clearly it is an $O(N^3)$ solver:

        python3 fredholm_numpy.py

Once we use PETSc KSP we have a chance to compare solvers.  First some $O(N^3)$ direct versions:

        python3 fredholm.py -ksp_view   # note ILU is a direct solver for dense Mats!

        python3 fredholm.py -mat_type dense -ksp_type preonly -pc_type lu  -pc_factor_mat_ordering_type natural

This iterative method should be $O(N^2)$ if the number of iterations is mesh independent, which would appear to be true:

        python3 fredholm.py -mat_type aij -ksp_type gmres -ksp_converged_reason -pc_type jacobi -ksp_rtol 1.0e-9

The last solver is headed toward being fastest at higher resolutions, but in fact the numpy method is still faster at the tested resolutions.
