# Particle-Swarm-System-Solver
Derivative-free nonlinear system solver with no initial guess. Attempts to solve F(x)=0 in x using a basic particle swarm algorithm (cf. https://en.wikipedia.org/wiki/Particle_swarm_optimization). F is a vector-valued function of vector argument. The solution is searched in the hyperrectangle given by lbound and ubound, i.e. it is supposed that lbound[i] &lt;= x[i] &lt;= ubound[i], i = 1,2,..,d.
