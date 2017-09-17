'''
Derivative-free nonlinear system solver with no initial guess.
Attempts to solve F(x)=0 in x using a basic particle swarm algorithm (cf. https://en.wikipedia.org/wiki/Particle_swarm_optimization)
F is a vector-valued function of vector argument
The solution is searched in the hyperrectangle given by lbound and ubound, i.e.
it is supposed that lbound[i] <= x[i] <= ubound[i], i = 1,2,..,d
    
A basic PSO algorithm is defined as follows. Let f be the objective function, (lbound, ubound) the boundaries of the search-space,
S be the number of particles in the swarm, each having a position xi in the search-space 
and a velocity vi. Let pi be the best known position of particle i and let g be the best known position of the entire swarm

for each particle i = 1, ..., S do
   Initialize the particle's position with a uniformly distributed random vector: xi ~ U(lbound, ubound)
   Initialize the particle's best known position to its initial position: pi <- xi
   if f(pi) < f(g) then
       update the swarm's best known  position: g <- pi
   Initialize the particle's velocity: vi ~ U(-|lbound-ubound|, |lbound-ubound|)
while a termination criterion is not met do:
   for each particle i = 1, ..., S do
      for each dimension d = 1, ..., n do
         Pick random numbers: rp, rg ~ U(0,1)
         Update the particle's velocity: vi[d] <- omega vi,d + c1 rp (pi[d]-xi[d]) + c2 rg (g[d]-xi[d])
      Update the particle's position: xi <- xi + vi
      if f(xi) < f(pi) then
         Update the particle's best known position: pi <- xi
         if f(pi) < f(g) then
            Update the swarm's best known position: g <- pi
        
omega, c1 and c2 are scalar parameters of the algorithm

Requirements: numpy
Developer: Ivan Gorynin
Contact Info: ivan.gorynin@aol.com
'''

import numpy as np

class ParticleSwarm:
    particles = []
    swarmBestParticle = None
    
    class Particle:
        __slots__ = ['bestResidual','bestPosition', 'position', 'velocity']
        bestResidual = None
        bestPosition = None
        def __init__(self, position, velocity):
            self.position = position
            self.bestPosition = position
            self.velocity = velocity
            self.bestResidual = float('Inf')
            
    def __init__(self, objectiveFunction, lbound, ubound, populationSize=100, omega = 0, c1 = 2, c2 = 2):
        self.lbound = lbound
        self.ubound = ubound
        self.populationSize = populationSize
        self.nvals = len(lbound)
        self.objectiveFunction = objectiveFunction;
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        
    def initializeSwarm(self):
        self.particles =(map(ParticleSwarm.Particle, \
            np.random.uniform(self.lbound, self.ubound, [self.populationSize, self.nvals]),\
            np.random.uniform(self.lbound-self.ubound, self.ubound-self.lbound, [self.populationSize, self.nvals])))
        for p in self.particles:
            p.bestResidual = self.objectiveFunction(p.position)
            if p.bestResidual<=self.getBestResidual():
                self.swarmBestParticle = p
        
    def makeIteration(self):
        for p in self.particles:
            while True:
                velocity = self.omega*p.velocity + self.c1*np.random.uniform(size=[self.nvals])*(p.bestPosition - p.position) \
                    + self.c2*np.random.uniform(size=[self.nvals])*(self.getBestPosition() - p.position)
                position = p.position + velocity
                if np.all(position>=self.lbound) and np.all(position<=self.ubound):
                    p.position = position
                    p.velocity = velocity
                    break

            residual = self.objectiveFunction(p.position)
            
            if residual<=p.bestResidual:
                p.bestResidual = residual
                p.bestPosition = p.position
    
            if p.bestResidual<=self.getBestResidual():
                self.swarmBestParticle = p
            
    def getBestResidual(self):
        if (self.swarmBestParticle is None):
            return float('Inf')
        else:
            return self.swarmBestParticle.bestResidual
        
    def getBestPosition(self):
        if (self.swarmBestParticle is None):
            return None
        else:
            return self.swarmBestParticle.bestPosition

def particleSwarmSolve(objectiveFunction, lbound, ubound, populationSize=100, omega = 0, c1 = 2, c2 = 2, maxIters = 2000, tolerance = 1E-5):
    objSwarm = ParticleSwarm(lambda x: np.max(np.absolute(objectiveFunction(x))), lbound, ubound, populationSize, omega, c1, c2)
    objSwarm.initializeSwarm()
    nbIterations = 0
    
    for i in range(maxIters):
        objSwarm.makeIteration()
        nbIterations+=1
        if (objSwarm.getBestResidual()<=tolerance):
            break
    isConverged = objSwarm.getBestResidual()<=tolerance
    return objSwarm.getBestPosition(), objSwarm.getBestResidual(), isConverged, nbIterations

def demonstrateSolver():
    '''
    Demonstates the effeciency of the solver applied to the following system having d unknowns:
        (3+2x_1)x_1 - 2x_2 = 3
        (3+2x_i)x_i - x_{i-1} - 2x_{i+1} = 2, i = 2,3,..,d-1
        (3+2x_d)x_d - x_{d-1} = 4
        
    This system is specified via function F, which arguemnt and output are objects of type 
    numpy.array
    It is supposed that 0 <= x[i] <= 2, i = 1,2,..,d. The exact solution of this system is
    x[i] = 1, i = 1,2,..,d
    '''
    d = 5
    
    def f(x):
        result = np.zeros(d, 'float')
        for i in range(1,d-1):
            result[i] = (3+2*x[i])*x[i] - x[i-1] - 2*x[i+1] - 2
        result[0] = (3+2*x[0])*x[0] - 2*x[1] - 3
        result[d-1] = (3+2*x[d-1])*x[d-1] - x[d-2] - 4
        return result
    
    lbound = np.zeros(d, 'float');
    ubound = 2*np.ones(d, 'float'); 

    x0, residual, isConverged, nbIterations = \
        particleSwarmSolve(f, lbound, ubound, c1=0, c2=2, tolerance=1E-12)

    if isConverged:
        print("The solver converged in " + str(nbIterations) + " iterations")
    else:
        print("The solver did not converge in " + str(nbIterations) + " iterations")
        
    print("x0:\t"+str(x0))
    print("Residual:\t"+str(residual))

demonstrateSolver()
