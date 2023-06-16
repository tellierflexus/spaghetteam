import trusspy as tp
import networkx as nx
import itertools
import math
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
#matplotlib widget
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("error")
import scipy
import pickle
from threading import Thread
from slientruss3d.truss import Truss
from slientruss3d.type  import SupportType, MemberType
import copy

MAX_LENGTH = 25
MAX_MARSH = 15


class Tower:

    def __init__(self, nbr_layers=4, min_marsh=3, max_marsh=20, min_radius=4, max_radius=40, layer_height=10, layers_data=None):
        if layers_data is None:
            self._layers = [{"nbr_marsh": round(np.random.uniform(low=min_marsh, high=max_marsh)), "radius": round(np.random.uniform(low=min_radius, high=max_radius))} for _ in range(nbr_layers)]
        else:
            self._layers = layers_data
        self._layer_height=layer_height
        self._egdges = []
        self._init_edges()

    @property
    def layers(self):
        #print(f"Number of layers : {len(self._layers)}")
        return self._layers

    def _init_edges(self):

        def get_length(coords_1, coords_2):
            return math.sqrt((coords_1[0]-coords_2[0])**2 + (coords_1[1]-coords_2[1])**2 + (coords_1[2]-coords_2[2])**2)

        marshmallows = self.get_nodes(layer=0)
        for num_layer in range(len(self._layers)):
            start_index = 0
            for k in range(num_layer):
                start_index += self._layers[k]["nbr_marsh"]

            
            for i in range(len(marshmallows)):
                for j in range(len(marshmallows)):
                    if i==j:
                        continue
                    if get_length(marshmallows[i], marshmallows[j]) < MAX_LENGTH:
                        edge = [start_index+i, start_index+j]
                        if j < i:
                            edge.reverse()
                        if edge not in self._egdges:
                            self._egdges.append(edge)
            if num_layer < len(self._layers) -1:
                marshmallows_next_layer = self.get_nodes(layer=num_layer+1)
                for i in range(len(marshmallows)):
                    for j in range(len(marshmallows_next_layer)):
                        if get_length(marshmallows[i], marshmallows_next_layer[j]) < MAX_LENGTH:
                            edge = [start_index+i, start_index+self._layers[num_layer]["nbr_marsh"]+j]
                            if edge not in self._egdges:
                                self._egdges.append(edge)
                marshmallows=marshmallows_next_layer

 
    @property
    def edges(self):
        return self._egdges

    @property
    def nbr_nodes(self):
        sum = 0
        for i in range(len(self._layers)):
            sum += self._layers[i]["nbr_marsh"]
        return sum
    
   
    def mutate(self, min_marsh=3, max_marsh=20, min_radius=4, max_radius=100):

        nbr_mutations = round(np.random.uniform(low=1, high=len(self._layers)))
        mutated_layers = list(range(len(self._layers)))
        np.random.shuffle(mutated_layers)
        mutated_layers = mutated_layers[:nbr_mutations]

        for layer in mutated_layers:
            self._layers[layer] = {"nbr_marsh": min(max(round(np.random.normal(self._layers[layer]["nbr_marsh"],2)), min_marsh), max_marsh), "radius": min(max(round(np.random.normal(self._layers[layer]["radius"],4)), min_radius), max_radius)}
        self._egdges = []
        self._init_edges()
            #print(f"Mutated layers : {mutated_layers}")
        return self


    def get_nodes(self, layer=None):

        def get_nodes_from_layer(layer):
            if layer < len(self._layers):
                self._layers[layer]["radius"]
                return [(self._layers[layer]["radius"]*np.cos(np.pi*2*i/self._layers[layer]["nbr_marsh"]), self._layers[layer]["radius"]*np.sin(np.pi*2*i/self._layers[layer]["nbr_marsh"]), layer*self._layer_height) for i in range(self._layers[layer]["nbr_marsh"])]
            else:
                return []

        if layer is not None:
            return get_nodes_from_layer(layer)
        else:
            nodes = []
            for i in range(len(self._layers)):
                nodes.extend(get_nodes_from_layer(i))
            return nodes

    

def draw_tower(tower):
    nodes = tower.get_nodes()
    x, y, z = zip(*nodes)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=z, cmap='seismic', linewidths=10, alpha=1)
    for edge in tower.edges:
        x_ed, y_ed, z_ed = zip(*[nodes[index] for index in edge]) #-1 because the index in edges start at 1 to be compliant with trusspy
        #x_ed, y_ed, z_ed = zip(*list(edge))
        ax.plot(x_ed,y_ed,z_ed, c="#FFBB00")
    plt.title(str(tower.layers))
    plt.show()
    fig.savefig("model_undeformed_inc0_3d.png")


def crossover(population, strategy="same layer"):
    new_population = []
    for i in range(len(population)):
        tower_a = population[i]
        tower_b = population[-1-i]
        match strategy:
            case "same layer":
                layers_data = []
                for j in range(len(tower_a.layers)):
                    if np.random.rand() > 0.5:
                        layers_data.append(tower_a.layers[j])
                    else:
                        layers_data.append(tower_b.layers[j])
                new_population.append(Tower(layers_data=layers_data))
            case _:
                print("Hello")
    return new_population



def create_next_generation(population, fitness_values, mode="elitism", size=10, elitism_size=2, crossover_size=4, mutation_size=4):
    new_population = []
    match mode:
        case "elitism":
            sorted_population = [individual for _, individual in sorted(zip(fitness_values, population), key=lambda pair: pair[0])]         
            new_population.extend(sorted_population[:elitism_size])
            sorted_population = sorted_population[elitism_size:]
            np.random.shuffle(sorted_population) #to prevent always the same element beeing crossover of mutate
            new_population.extend(crossover(sorted_population[:crossover_size]))

            new_population.extend([tower.mutate() for tower in sorted_population[crossover_size:]])
            thread_results = []
            threads = [None] * len(new_population)
            for i in range(len(threads)):
                threads[i] = Thread(target=fitness, args=(new_population[i], thread_results))
                threads[i].start()
                #print(f"Started thread {i}")
            
            for i in range(len(threads)):
                threads[i].join()
            return zip(*thread_results)
            #return new_population, [fitness(tower) for tower in new_population]

        case "fusion":

            new_population.extend(population)
            for tower in population:
                new_tower = copy.deepcopy(tower)
                new_tower.mutate()
                new_population.append(new_tower)

            new_population.extend(crossover(population))
            thread_results = []
            threads = [None] * len(new_population)
            print(f"population len : {len(new_population)}")
            for i in range(len(new_population)):
                threads[i] = Thread(target=fitness, args=(new_population[i], thread_results))
                threads[i].start()
                #print(f"Started thread {i}")
            
            for i in range(len(threads)):
                threads[i].join()
            sorted_population = [(individual, fitness) for individual, fitness in sorted(thread_results, key=lambda pair: pair[1])]
            #for i in sorted_population:
            #    print(i[1])
            print(f"Best fitness : {sorted_population[0][1]}")
            #print("using fusion mode")
            return zip(*sorted_population[:size])



def fitness(tower, results, log=False):
    truss = Truss(dim=3)
    total_ext_forces =0
    nodes = tower.get_nodes()
    for i in range(len(nodes)):
        if nodes[i][2]==0:
            support=SupportType.ROLLER_Z
        else:
            support=SupportType.NO
        truss.AddNewJoint(nodes[i], support)
        truss.AddExternalForce(i, (0, 0, -20))
        if nodes[i][2]==nodes[-1][2]:
            truss.AddExternalForce(i, (0, 0, -100))
            total_ext_forces+=1

    edges = tower.edges
    memberType = MemberType(1, 1e7, 1e5)
    for edge in edges:
        truss.AddNewMember(edge[0], edge[1], memberType)
    #print("total ext forces : "  + str(total_ext_forces))
    trials=[]
    for _ in range(5):
        try:
            truss.Solve()
        except Exception as e:
            #print(str(e))
            #trials.append(np.inf)
            print(str(e))
        else:
            forces = truss.GetInternalForces(isProtect=True)
            trials.append(max(forces.values()))
    
    if len(trials)>0:
        force = np.mean(trials)
    else:
        force = np.inf
    

    #fitness = (len(tower.edges)+tower.nbr_nodes)*max(forces.values())/total_ext_forces
    #fitness = (len(tower.edges)+tower.nbr_nodes)*force/total_ext_forces
    fitness = (len(tower.edges)+tower.nbr_nodes)*force/total_ext_forces
    results.append((tower, fitness))



if __name__ == "__main__":
    """
    results_after_iterations = []
    fitnesses = [np.inf]
    while (fitnesses[0] == np.inf):
        tower = Tower()
        results = []
        fitness(tower, results)
        _, fitnesses = zip(*results)
    input("Press Enter to continue...")
    for _ in range(30):
        results = []
        fitness(tower, results)
        _, fitnesses = zip(*results)   
        results_after_iterations.append(fitnesses[0]) 
    plt.scatter(list(range(len(results_after_iterations))), results_after_iterations)
    plt.show()

    """
    f = open("fitnesses.txt", "w")
    
    population = [Tower() for _ in range(10)]
    results = []


    for i in range(len(population)):

        fitness(population[i], results)
    _, fitnesses = zip(*results)


    generation = 0
    mean_fitness = []
    min_fitness = []
    print(f"Generation {generation}, length : {len(population)}, {len(fitnesses)} , Fitnesses : {sorted(fitnesses)}")
    f.write(f"Generation {generation}, Fitnesses : {sorted(fitnesses)}\n")
    mean_fitness.append(np.mean(fitnesses))
    progress = 1
    while (progress):
        population, fitnesses = create_next_generation(population, fitnesses, mode="fusion", size=10)
        generation+=1
        print(f"Generation {generation}, length : {len(population)} , Fitnesses : {sorted(fitnesses)}")
        f.write(f"Generation {generation}, Fitnesses : {sorted(fitnesses)}\n")
        mean_fitness.append(np.mean(fitnesses))
        min_fitness.append(np.min(fitnesses))
        if (len(min_fitness) >=5 ) and (min_fitness[-5] - np.min(fitnesses) < 10):
            progress=0
        #input("Press Enter to continue...")
    draw_tower(population[fitnesses.index(min(fitnesses))])
    plt.scatter(list(range(len(mean_fitness))), mean_fitness)
    plt.xlabel('generations')
    plt.ylabel('mean fitness function value')
    plt.show()
    f.close()
    with open('towers.pkl', 'wb') as out_file:
        pickle.dump(population, out_file)
    

    
    



