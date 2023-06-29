# drake-integral-eqn

The [Firedrake](https://www.firedrakeproject.org) finite element library is usually used to discretize the weak forms of partial differential equations (PDEs), but here we want to demonstrate how to adapt it to integral equations.  This is not completely natural!

For a start we consider a linear integral equation and merely create a numpy linear system $Ac=b$ and call `numpy.linalg.solve()` on it.

## example

A well-understood type of integral equation is
  $$u(x) + \int_0^1 K(x,y) u(y) dy = f(x)$$
where we seek $u \in L^2(0,1)$ for given functions $K \in C([0,1]^2)$ and $f\in L^2(0,1)$.  The weak form of the above equation is
  $$\int_0^1 u(x) v(x) dx + \int_0^1 \int_0^1 K(x,y) u(y) v(x) dy dx = \int_0^1 f(x) v(x) dx$$
We can assemble a linear system $Ac=b$ column-by-column if we expand $u$ in basis (hat) functions,
  $$u(x) = \sum_{j=0}^{N} c_j \phi_j(x)$$
Inserting this into the weak form and using $v(x)=\phi_i(x)$ we get an entry of the matrix $A$:
  $$a_{ij} = \int_0^1 \phi_j(x) \phi_i(x) dx + \int_0^1 \left(\int_0^1 K(x,y) \phi_j(y) dy\right) \phi_i(x) dx$$
Also
  $$b_i = \int_0^1 f(x) \phi_i(x) dx$$
FIXME continue