import numpy as np
from rbfn import computeScore

class PSO:
    def __init__(self, objective_func, DNA, max_iter, Vmin, Vmax):
        self.objective_func = objective_func
        self.num_particles = DNA.shape[0]
        self.num_dimensions = DNA.shape[1]
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.max_iter = max_iter
        self.particles_position = DNA
        self.particles_velocity = np.zeros(DNA.shape)
        self.particles_best_position = self.particles_position.copy()
        self.particles_best_value = np.zeros(self.num_particles)
        self.global_best_position = np.zeros(self.num_dimensions)
        self.global_best_value = float('inf')

    def optimize(self):
        for i in range(self.max_iter):
            for j in range(self.num_particles):
                self.particles_best_value[j] = self.objective_func(self.particles_best_position[j])

            for j in range(self.num_particles):
                if self.particles_best_value[j] < self.global_best_value:
                    self.global_best_value = self.particles_best_value[j]
                    self.global_best_position = self.particles_best_position[j].copy()

            for j in range(self.num_particles):
                r1 = np.random.random(self.num_dimensions)
                r2 = np.random.random(self.num_dimensions)
                self.particles_velocity[j] = self.particles_velocity[j] + 2.0 * r1 * (self.particles_best_position[j] - self.particles_position[j]) \
                                            + 2.0 * r2 * (self.global_best_position - self.particles_position[j])
                
                self.particles_velocity[j] = np.clip(self.particles_velocity[j], self.Vmin, self.Vmax)
                self.particles_position[j] = self.particles_position[j] + self.particles_velocity[j]

                


        return self.global_best_position, self.global_best_value

def fitness_func(individual):
    return computeScore(individual)

def getbestDNA(DNA, max_iter, Vmin, Vmax):
    pso = PSO(fitness_func, DNA, max_iter, Vmin, Vmax)
    best_position, best_value = pso.optimize()
    print("best position:")
    print(best_position)
    return best_position

"""
# 示例使用
def objective_function(x):
    return np.sum(np.square(x))

DNA = np.array([[1, 1], [2, 2]])

pso = PSO(objective_function, DNA, 100)
best_position, best_value = pso.optimize()
print("最优位置：", best_position)
print("最优值：", best_value)
"""