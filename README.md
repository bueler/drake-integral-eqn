# drake-integral-eqn

The [Firedrake](https://www.firedrakeproject.org) finite element library is usually used to discretize the weak forms of partial differential equations (PDEs), but here we want to experiment with integral equations.  This is a bit awkward.

For now we consider only linear integral equations.  The basic strategy is to use Firedrake tools to assemble as many parts of the discrete linear system as possible.  The matrix corresponding to the integral operator would seem not to permit a single `assemble()` command.  Instead it is assembled column-by-column in a somewhat "by-hand" manner.  There may well be a better approach.

TODO:

  * 2D example
  * what would Brandt & Lubrecht (1990) do?

## example

A well-understood type of linear integral equation is a Fredholm "second kind" equation:
  $$u(x) + \int_0^1 K(x,y) u(y) dy = f(x)$$
We seek $u \in L^2(0,1)$ for given functions $K \in C([0,1]^2)$ and $f\in L^2(0,1)$.

The weak form of the above equation is
  $$\int_0^1 u(x) v(x) dx + \int_0^1 \int_0^1 K(x,y) u(y) v(x) dy dx = \int_0^1 f(x) v(x) dx$$
The first integral is the well-known mass matrix, which has obvious Firedrake assembly:

        V = FunctionSpace(mesh, 'CG', 1)  # for example
        u = TrialFunction(V)
        v = TestFunction(V)
        Mass = assemble(u * v * dx)

Likewise, the right-hand side of the weak form is standard:

        x = SpatialCoordinate(mesh)[0]
        b = assemble(f(x) * v * dx)

For the integral operator weak form, however, no Firedrake assembly expression is apparent.[^1]

Instead we assemble a matrix column-by-column by expanding the trial function argument.  That is, we expand $u$ in basis (hat) functions:
  $$u(x) = \sum c_j \phi_j(x)$$
where $\phi_j(x)$ is the hat function at $x_j$.  Inserting this gives a one-form which corresponds to a column of the matrix $K=$ `Kmat` below:
  $$K_{:,j} = \int_0^1 \tilde K_j(x) v(x) dx$$
where
  $$\tilde K_j(x) \approx \int_0^1 K(x,y) \phi_j(y) dy$$
The code we have in mind is this:

        phij = Function(V)
        Y = Constant(0.0)
        for j, xj in enumerate(V.mesh().coordinates.dat.data):
            phij.dat.data[:] = 0.0
            phij.dat.data[j] = 1.0
            Y.assign(xj)
            Kmat.setValues(range(V.dim()),[j,],
                           assemble(c * K(x,Y) * v * dx).dat.data)

Here the constant `c` is the mesh spacing in 1D, and generally it is the volume of a cell centered at $x_j$, _a la_ finite volumes.  That is, we are using a kind of weighted midpoint rule:
  $$\tilde K_j(x) = h_x K(x,x_j)$$
(This formula is modified at the boundary.)

Then the system matrix is set up and solved, here with KSP as we do in `fredholm.py`:

        Amat = Mass.M.handle + Kmat
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setOperators(A=Amat, P=Amat)
        ksp.setFromOptions()
        u = Function(V)
        with u.dat.vec as vu:
            with b.dat.vec_ro as vb:
                ksp.solve(vb, vu)

## demo solvers

Consider this non-symmetric kernel case:

  $$\begin{align*}
    K(x,y) &= - 0.5 (x + 1) e^{-x y} \\
    f(x) &= e^{-x} - 0.5 + 0.5 e^{-(x+1)}
    \end{align*}$$

The exact solution is $u(x)=e^{-x}$.

We have two solvers:

  * `fredholm.py`: Implements the above approach, including the KSP solver usage.

  * `fredholm_numpy.py`:  In this naive approach we turn everything into [numpy](https://numpy.org/) arrays and call `numpy.linalg.solve()` on it.  This approach is actually quite effective because everything is dense.  However, it is inflexible with respect to the solver.

## running the demos

> **Warning**  
> At high resolutions these demos should be run with a process/system monitor on top so bad cases can be killed.  That is, kill processes if you either don't want to wait for completion or you are running out of memory.  Also, reported timings include cache times; re-run for "real" solve times.

Here is the naive numpy approach with $N=1024$ elements, which actually has good performance because the matrices are dense.  Clearly it is an $O(N^3)$ solver:

        python3 fredholm_numpy.py 1024

Once we use PETSc KSP we have a chance to compare solvers.  First some $O(N^3)$ direct versions:

        python3 fredholm.py 1024 -ksp_view

(This is direct because ILU is a direct solver for dense matrices!)

If we deliberately want LU:[^2]

        python3 fredholm.py 1024 -mat_type dense -ksp_type preonly -pc_type lu -pc_factor_mat_ordering_type natural

The following iterative methods should be $O(ZN^2)$ if the number of iterations $Z$ is mesh independent, which indeed appears to be true!:

        python3 fredholm.py 1024 -mat_type aij -ksp_type gmres -ksp_converged_reason -pc_type jacobi -ksp_rtol 1.0e-9
        python3 fredholm.py 1024 -mat_type aij -ksp_type bcgs -ksp_converged_reason -pc_type jacobi -ksp_rtol 1.0e-9

(Also `-pc_type kaczmarz` makes sense. :smile:)  These are perhaps headed toward being fastest at super high resolutions, but in fact the numpy method is still faster at the _tested_ resolutions; e.g. compare with $N=8192$.

[^1]:  This [Fenics example](https://fenicsproject.org/qa/9537/assembling-integral-operators/) takes essentially our approach here.  An [old post by Anders Logg](https://answers.launchpad.net/dolfin/+question/141904) suggests a more direct approach is not possible.

[^2]:  `-pc_factor_mat_ordering_type natural` also avoids an apparent PETSc bug with the default `nd` ordering.
