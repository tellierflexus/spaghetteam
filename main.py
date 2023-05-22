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

MAX_LENGTH = 25


class Tower:

    def __init__(self, nbr_layers=4, min_marsh=3, max_marsh=10, min_radius=10, max_radius=40, layer_height=10, layers_data=None):
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
                        edge = [start_index+i+1, start_index+j+1]
                        if j < i:
                            edge.reverse()
                        if edge not in self._egdges:
                            self._egdges.append(edge)
            if num_layer < len(self._layers) -1:
                marshmallows_next_layer = self.get_nodes(layer=num_layer+1)
                for i in range(len(marshmallows)):
                    for j in range(len(marshmallows_next_layer)):
                        if get_length(marshmallows[i], marshmallows_next_layer[j]) < MAX_LENGTH:
                            edge = [start_index+i+1, start_index+self._layers[num_layer]["nbr_marsh"]+j+1]
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
    
   
    def mutate(self, min_marsh=3, max_marsh=10, min_radius=10, max_radius=40):
        nbr_mutations = round(np.random.uniform(low=1, high=len(self._layers)))
        mutated_layers = list(range(len(self._layers)))
        np.random.shuffle(mutated_layers)
        mutated_layers = mutated_layers[:nbr_mutations]

        for layer in mutated_layers:
            self._layers[layer] = {"nbr_marsh": round(np.random.uniform(low=min_marsh, high=max_marsh)), "radius": round(np.random.uniform(low=min_radius, high=max_radius))}
        self._egdges = []
        self._init_edges()
        print(f"Mutated layers : {mutated_layers}")
        return self

    def get_nodes(self, layer=None):

        def get_nodes_from_layer(layer):
            if layer < len(self._layers):
                self._layers[layer]["radius"]
                return [[self._layers[layer]["radius"]*np.cos(np.pi*2*i/self._layers[layer]["nbr_marsh"]), self._layers[layer]["radius"]*np.sin(np.pi*2*i/self._layers[layer]["nbr_marsh"]), layer*self._layer_height] for i in range(self._layers[layer]["nbr_marsh"])]
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
        x_ed, y_ed, z_ed = zip(*[nodes[index-1] for index in edge]) #-1 because the index in edges start at 1 to be compliant with trusspy
        #x_ed, y_ed, z_ed = zip(*list(edge))
        ax.plot(x_ed,y_ed,z_ed, c="#FFBB00")
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



def create_next_generation(population, fitness_values, mode="elitism", size=10, elitism_size=2, crossover_size=6, mutation_size=2):
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
                print(f"Started thread {i}")
            
            for i in range(len(threads)):
                threads[i].join()
            return zip(*thread_results)
            #return new_population, [fitness(tower) for tower in new_population]

        case "default":
            return population


def fitness(tower, results, log=False):
    print("eeeeh zzzzzéééé parti")
    M = tp.Model(logfile=log)
    with M.Nodes as MN:
        nodes = tower.get_nodes()
        for i in range(len(nodes)):
            MN.add_node(i+1, coord=nodes[i])
    element_type   = 1    # truss
    material_type  = 1    # linear-elastic

    youngs_modulus = 4*10**9
    cross_section_area = math.pi*(1.25*10**(-3))**2

    with M.Elements as ME:
        edges = tower.edges
        for i in range(len(edges)):
            ME.add_element( i+1, conn=edges[i] )
        ME.assign_etype("all", element_type)
        ME.assign_mtype("all", material_type)
        ME.assign_material("all", [youngs_modulus])
        ME.assign_geometry("all", [cross_section_area])

    with M.Boundaries as MB:
        for i in range(len(nodes)):
            if nodes[i][2] != 0:
                break
            MB.add_bound_U( i+1, (1,1,0) )

    total_ext_forces = 0
    with M.ExtForces as MF:
        for i in range(len(nodes)):
            if nodes[i][2] != nodes[-1][2]:
                continue
            MF.add_force( i+1, ( 0, 0,-1) )
            total_ext_forces += 1

    M.Settings.dlpf = 0.005
    M.Settings.du = 0.05
    M.Settings.incs = 163
    M.Settings.stepcontrol = False
    M.Settings.maxfac = 4
    
    M.Settings.ftol = 8
    M.Settings.xtol = 8
    M.Settings.nfev = 8
    
    M.Settings.dxtol = 1.25
    try:
        M.build()
    except ValueError as e:
        results.append((tower, np.inf))
        return
    try:
        M.run()
    except RuntimeWarning:
        results.append((tower, np.inf))
        return
    except ValueError as e:
        results.append((tower, np.inf))
        return
    except scipy.sparse.linalg._dsolve.linsolve.MatrixRankWarning as e:
        results.append((tower, np.inf))
        return
    except Exception as e:
        results.append((tower, np.inf))
        return
        
    print("total ext forces : "  + str(total_ext_forces))
    #draw_tower(tower)
    #pinc = 40  # 105
    #fig, ax = M.plot_model(
    #    view="3d",
    #    contour="force",
    #    lim_scale=(-10, 10, -10, 10, 0, 20),  # 3d
    #    force_scale=0.4*10e5,
    #    inc=-1,
    #)
    #plt.show()
    #fitness = len(tower.edges)*max(M.Results.R[-1].element_force)[0]*10e3/total_ext_forces
    fitness = tower.nbr_nodes*max(M.Results.R[-1].element_force)[0]*10e3/total_ext_forces
    results.append((tower, fitness))
    #return fitness




if __name__ == "__main__":
    population = [Tower() for _ in range(10)]
    results = []
    a = 0
    for i in range(len(population)):
        a+=1
        fitness(population[i], results)
    _, fitnesses = zip(*results)
    generation = 0
    print(f"Generation {generation}, length : {len(population)}, {len(fitnesses)} , Fitnesses : {sorted(fitnesses)}")
    input("Press Enter to continue...")
    
    while generation < 10:
        population, fitnesses = create_next_generation(population, fitnesses)
        generation+=1
        print(f"Generation {generation}, length : {len(population)} , Fitnesses : {sorted(fitnesses)}")
        #input("Press Enter to continue...")
    draw_tower(population[fitnesses.index(min(fitnesses))])
    with open('towers.pkl', 'wb') as out_file:
        pickle.dump(population, out_file)
    

    
    



