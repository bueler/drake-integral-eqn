# drake-integral-eqn

The [Firedrake]() finite element library is usually used to discretize the weak forms of partial differential equations (PDEs), but here we adapt it to integral equations.

## example

For example, a well-understood type of integral equation is

$$u(x) + \int_0^1 K(x,y) u(y) dy = f(x)$$

where we seek $u \in L^2(0,1)$ for given functions $K \in L^\infty(0,1)^2$ and $f\in L^2(0,1)$.
