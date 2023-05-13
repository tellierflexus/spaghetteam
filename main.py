from numpy import random
import itertools 
import math
import networkx as nx

import matplotlib.pyplot as plt
#matplotlib widget
from mpl_toolkits.mplot3d import Axes3D

MAX_LENGTH = 4
SQUARE_SIZE=4
LAYER_HEIGHT=5
MUTATION_ADD=5

def create_individual(nbr_layers, mu=6, sigma=0.7):
    """
    Create an individual 
    An individual is a list of 4 layer, each layer is a list containing tuple for representing marhsmallow position
    example : 
    one_individual = [ 
    [(1,1), (2,3), (5,6)], 
    [(1,1), (2,3), (5,6), (2,3), (5,6)], 
    [(1,1), (2,3), (5,6), (2,3), (5,6), (2,3), (5,6)], 
    [(1,1), (2,3)] 
    ]


    """
    individual = [[ (round(SQUARE_SIZE*random.normal(0.5,0.2)) , round(SQUARE_SIZE*random.normal(0.5,0.2)), layer*LAYER_HEIGHT) for _ in range(round(random.normal(mu,sigma)))] for layer in range(nbr_layers)]
    return individual


def create_initial_population(nbr_ind, nbr_layers, mu=5, sigma=1):
    """
    Generate an initial population of spaghetti marshmallow towers

    Input : 
    nbr_ind : the number of towers (individuals) in this inital generation
    nbr_layers : the number of layers in each tower
    mu : Mean for gaussian distribution of number of marshmallow in each layer
    sigma : Sigma for gaussian distribution of number of marshmallow in each layer
    
    Output:
    A list of individual
    An individual is a list of layers, each layer is a list containing tuple for representing marhsmallow position
    
    
    """
    population = [  create_individual(nbr_layers, mu, sigma)   for _ in range(nbr_ind) ]
    return population


def generate_graph(individual):
    """
    Generate a networkx graph from an individual

    Input:
    An individual

    Output :
    A networkx graph 
    """
    G = nx.Graph()
    nbr_layers = len(individual)
    for layer in range(nbr_layers):
        G.add_nodes_from(individual[layer]) #we add this layer to the graph 
        potential_spaghetti = list(itertools.combinations(individual[layer],2))
        if layer < nbr_layers - 1:
            potential_spaghetti += list(itertools.product(individual[layer], individual[layer+1])) #we add spaghetti between levels


        for potential_spaghetto in potential_spaghetti:
            length = math.sqrt((potential_spaghetto[0][0] - potential_spaghetto[1][0])**2 + (potential_spaghetto[0][1] - potential_spaghetto[1][1])**2)
            if length < MAX_LENGTH:
                G.add_edge(potential_spaghetto[0], potential_spaghetto[1])

    return G





def check_connectivity(graph):
    """
    Check if there is only one tower, i.e from any marshallow we can reach any other marshallow, i.e., the graph is connected. 

    Input:
    a graph corresponding to an individual

    Output:
    The degree of connectivity, i.e. the smallest number of connection a marshmallow have
    """     

    degree_sequence = (d for n, d in graph.degree())
    return min(degree_sequence)


def draw_tower(graph):
    x, y, z = zip(*list(graph.nodes()))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=z, cmap='seismic', linewidths=10, alpha=1)
    for edge in list(graph.edges()):
        x_ed, y_ed, z_ed = zip(*list(edge))
        ax.plot(x_ed,y_ed,z_ed, c="#FFBB00")
    plt.show()
    fig.savefig("model_undeformed_inc0_3d.png")


def crossover(population):
    new_population = []
    for index in range(len(population)):
        ind_a = population[index]
        ind_b = population[-index-1]
        new_individual_a = []
        new_individual_b = []
        for layer in range(len(ind_a)):
            new_layer_a = []
            new_layer_b = []
            new_layer_a.extend(ind_a[layer][:round(random.random()*(len(ind_a[layer])-1))])
            new_layer_a.extend(ind_b[layer][round(random.random()*(len(ind_b[layer])-1)):])
            new_layer_b.extend(ind_a[layer][round(random.random()*(len(ind_a[layer])-1)):])
            new_layer_b.extend(ind_b[layer][:round(random.random()*(len(ind_b[layer])-1))])
            new_individual_a.append(new_layer_a)
            new_individual_b.append(new_layer_b)
        new_population.append(new_individual_a)
        new_population.append(new_individual_b)
    return new_population

def mutate(population):
    mutated_population = []
    print("size of ind 0 layer 0 " + str(len(population[0][0])))
    for ind in population:
        for layer in range(len(ind)):
            for _ in range(round(MUTATION_ADD*random.random())):
                ind[layer].append((round(SQUARE_SIZE*random.random()), round(SQUARE_SIZE*random.random()), layer))
                print("added marshmallow to layer " + str(layer))
        mutated_population.append(ind)
    print("size of ind 0 layer 0 after mutation " + str(len(population[0][0])))
    return mutated_population


    return mutated_population

def create_next_generation(population, fitness_values, mode="elitism", size=10, elitism_size=2, crossover_size=6, mutation_size=2):
    new_population = []
    match mode:
        case "elitism":
            sorted_population = [individual for _, individual in sorted(zip(fitness_values, population), key=lambda pair: pair[0])]         
            new_population.extend(sorted_population[:elitism_size])
            sorted_population = sorted_population[elitism_size:]
            random.shuffle(sorted_population) #to prevent always the same element beeing crossover of mutate
            new_population.extend(crossover(sorted_population[:crossover_size]))
            new_population.extend(mutate(sorted_population[crossover_size:]))
            return new_population
        case "default":
            return population

if __name__ == "__main__":
    #a = create_individual(4)
    #graph = generate_graph(a)
    #print(graph.nodes())
    #print(graph.edges())
    #print(check_connectivity(graph))
    #draw_tower(graph)
    population = create_initial_population(10,4)
    fitness_values = [random.random() for _ in range(len(population))]
    new_pop = create_next_generation(population, fitness_values)
    print(len(new_pop))
    #print(new_pop[0])
    graph = generate_graph(new_pop[0])
    draw_tower(graph)

