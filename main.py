from numpy import random
import itertools 
import math
import networkx as nx


MAX_LENGTH = 5
SQUARE_SIZE=4

def create_individual(nbr_layers, mu=5, sigma=1):
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
    individual = [[ (round(SQUARE_SIZE*random.random()) , round(SQUARE_SIZE*random.random())) for _ in range(round(random.normal(mu,sigma)))] for _ in range(nbr_layers)]
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
    population = [  create_individual(nbr_layers, mu, sigma)   for _ in range(nbr_ind)   ]


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
        potential_spaghetti = itertools.combinations(individual[layer],2)
        if layer < nbr_layers - 1:
            potential_spaghetti.append(itertools.product(individual[layer], individual[layer+1])) #we add spaghetti between levels

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


if __name__ == "__main__":
    a = create_individual(1)
    print(a)
    graph = generate_graph(a)
    print(graph.nodes())
    print(graph.edges())
    print(check_connectivity(graph))