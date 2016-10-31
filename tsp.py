from __future__ import print_function
from parser import parse
from matplotlib import pyplot as plt
import math
import random
import copy

DATA_WESTERN_SAHARA = 'data/wi29.tsp'
DATA_DJIBOUTI = 'data/dj38.tsp'
DATA_QATAR = 'data/qa194.tsp'
DATA_URUGUAY = 'data/uy734.tsp'

EXPONENTIAL_DECAY = 'exponential_decay'
LINEAR_DECAY = 'linear_decay'
STATIC_DECAY = 'static_decay'


class Neuron:

    def __init__(self, id, x_pos, y_pos):
        self.id = id
        self.x_pos = x_pos
        self.y_pos = y_pos

    def get_coord(self):
        coord = (self.x_pos, self.y_pos)
        return coord

    def show_data(self):
        print("ID:", self.id, "X:", self.x_pos, "Y:", self.y_pos)

    def get_real_coord(self, scale):
        coord = (self.x_pos * scale, self.y_pos * scale)
        return coord

def get_distance(city1, city2):
    """
    Given tuples of (x,y) coordinates, compute distance between cities
    c = sqrt(a^2 + b^2)
    """
    # TODO: REmove max and min
    a = max(city1[0], city2[0]) - min(city1[0], city2[0])
    b = max(city1[1], city2[1]) - min(city1[1], city2[1])
    return math.sqrt(a ** 2 + b ** 2)


def normalize_coordinates(city_data):
    """
    Scale coordinates to the range [0, 1], centered around 0.5
    :param city_data:
    :return:
    """
    normalized_city_coordinates = {}

    min_x, max_x, min_y, max_y = get_max_and_min(city_data)

    # Centering
    centering_offset_x = max_x - (max_x - min_x) / 2
    centering_offset_y = max_y - (max_y - min_y) / 2

    # Scaling
    scale = max(max_x - min_x, max_y - min_y)

    for city in city_data:
        normalized_city_coordinates[city] = (
            (city_data[city][0] - centering_offset_x) / scale + 0.5,
            (city_data[city][1] - centering_offset_y) / scale + 0.5
        )

    return normalized_city_coordinates, scale


def get_closest_neuron(neurons, city_position, non_selectables=[]):
    """
    Find neuron closest to given city coordinates
    :param neurons: list of neurons
    :param city_position: (x,y) tuple
    :return: neuron and distance
    """

    distances = []  # Distance to each neuron
    for n in neurons:  # type: Neuron
        if n.id not in non_selectables:
            distances.append(get_distance(city_position, n.get_coord()))
        else:
            distances.append(float("inf"))

    min_distance = min(distances)
    bmu = neurons[distances.index(min_distance)]
    return bmu


def plot_map(city_data, neuron_list=[], save=False, filename=''):
    plt.clf()
    for city in city_data:
        plt.scatter(city_data[city][0], city_data[city][1])

    neuron_x = [n.x_pos for n in neuron_list]
    neuron_y = [n.y_pos for n in neuron_list]

    plt.plot(neuron_x, neuron_y, 'or-')
    plt.plot([neuron_list[0].x_pos, neuron_list[-1].x_pos], [neuron_list[0].y_pos, neuron_list[-1].y_pos], 'or-')

    if save:
        plt.savefig(filename)
    else:
        plt.show()


# def plot_map_rotated(city_data, neuron_list=[], save=False, filename=''):
#     plt.clf()
#     for city in city_data:
#         plt.scatter(city_data[city][1], 1 - city_data[city][0])
#
#     neuron_x = [n.x_pos for n in neuron_list]
#     neuron_y = [n.y_pos for n in neuron_list]
#
#     plt.plot(neuron_y, neuron_x, 'or-')
#     plt.plot([neuron_list[0].y_pos, neuron_list[-1].y_pos], [neuron_list[0].x_pos, neuron_list[-1].x_pos], 'or-')
#
#     if save:
#         plt.savefig(filename)
#     else:
#         plt.show()

def initialize_neurons(number_of_neurons, normalized_city_data):
    min_x = min(normalized_city_data.values(), key=lambda v: v[0])[0]
    min_y = min(normalized_city_data.values(), key=lambda v: v[1])[1]
    max_x = max(normalized_city_data.values(), key=lambda v: v[0])[0]
    max_y = max(normalized_city_data.values(), key=lambda v: v[1])[1]

    # TODO: FIX SCALING
    scale = max(max_x-min_x, max_y-min_y)/4
    neurons = [Neuron( i, math.cos(2 * i * math.pi / number_of_neurons ) * scale + 0.5,
                       math.sin(2 * i * math.pi / number_of_neurons) * scale + 0.3)
               for i in range(number_of_neurons)]

    return neurons


def get_max_and_min(city_data):
    """
    :param city_data: Dictionary of (x,y) tuples
    :return: max and min for x and y
    """

    min_x = min(city_data.values(), key=lambda v: v[0])[0]
    max_x = max(city_data.values(), key=lambda v: v[0])[0]
    min_y = min(city_data.values(), key=lambda v: v[1])[1]
    max_y = max(city_data.values(), key=lambda v: v[1])[1]

    return min_x, max_x, min_y, max_y


def update_neuron_weights(bmu, city_pos, neurons, learning_rate, neighborbood_radius):
    """
    Update the position of the bmu neuron and neighbors to move closer
    :param city_pos: (x,y) of city
    :param bmu:
    :param learning_rate:
    :param neighborbood_radius: total number of neighbors composed of adjacency along both sides of ring.

    :type bmu: Neuron
    :return:
    """
    # Update bmu weights
    bmu.x_pos += learning_rate * (city_pos[0] - bmu.x_pos)
    bmu.y_pos += learning_rate * (city_pos[1] - bmu.y_pos)

    # Update neighbor weights
    for i in range(1, neighborbood_radius+1):
        # Right side
        neigh_id_right = (bmu.id + i) % len(neurons)
        update_neighbor(neurons[neigh_id_right], bmu, city_pos, learning_rate, neighborbood_radius)

        # Left side
        neigh_id_left = (bmu.id - i) % len(neurons)
        update_neighbor(neurons[neigh_id_right], bmu, city_pos, learning_rate, neighborbood_radius)


def update_neighbor(neighbor, bmu, city_pos, learning_rate, neighborhood_radius):
    theta = neighborhood_function(bmu, neighbor, neighborhood_radius)
    neighbor.x_pos += theta * learning_rate * (city_pos[0] - neighbor.x_pos)
    neighbor.y_pos += theta * learning_rate * (city_pos[1] - neighbor.y_pos)


def neighborhood_function(bmu, neighbor, neighborhood_radius):
    """
    Magnitude of update
    :type bmu: Neuron
    :type neighbor: Neuron
    :return:
    """
    distance = get_distance(bmu.get_coord(), neighbor.get_coord())
    return math.exp(-distance**2 / (2 * neighborhood_radius ** 2))


def get_tour(cities, neurons, scale):
    tour_neurons = copy.deepcopy(neurons)
    non_selectables = []
    for city_coord in cities.values():
        closest_neuron = get_closest_neuron(tour_neurons, city_coord, non_selectables)
        non_selectables.append(closest_neuron.id)
        closest_neuron.x_pos = city_coord[0]
        closest_neuron.y_pos = city_coord[1]

    distance = 0
    for i in range(len(tour_neurons)-1):
        distance += get_distance(tour_neurons[i].get_real_coord(scale), tour_neurons[i+1].get_real_coord(scale))
    distance += get_distance(tour_neurons[-1].get_real_coord(scale), tour_neurons[0].get_real_coord(scale))

    return distance, tour_neurons


def tsp_som(cities_datafile, number_of_iterations, plot_interval, decay_type, number_of_neurons, learning_rate=0.1, neighborhood_radius_fraction=0.1):
    """

    :param cities_datafile:
    :param number_of_iterations:
    :param decay_type: 'static', 'linear' or 'exponentital'
    :param learning_rate:
    :param neighborhood_radius_fraction: fraction of neurons used as neighbors initially (each direction in ring)
    :return:
    """
    # Dictionary containing tuples of (x,y) coordinates
    # Key=numbering from 0 and up. Value = (x,y)
    cities = parse(cities_datafile)
    norm_cities, norm_scale = normalize_coordinates(cities)  # Dictionary containing normalized (x,y) coordinates
    neurons = initialize_neurons(number_of_neurons, norm_cities)
    init_neighborhood_radius = neighborhood_radius = len(cities) * neighborhood_radius_fraction
    init_learning_rate = learning_rate

    exponential_decay_constant = number_of_iterations / math.log(neighborhood_radius)
    nr_linear_decay_constant = init_neighborhood_radius/number_of_iterations
    lr_linear_decay_constant = init_learning_rate/number_of_iterations

    init_distance, init_tour = get_tour(norm_cities, neurons, norm_scale)

    ####Initial statistics####
    print("\nInitialization")
    print("Neighborhood radius:", neighborhood_radius)
    print("Number of neurons:", number_of_neurons)
    print("Initial tour distance:", init_distance)
    ##########################

    # plot_map(norm_cities, neurons, save=False)
    # plot_map(norm_cities, init_tour)  # Initial city-neuron allocation

    for i in range(number_of_iterations):
        chosen_city = random.choice(norm_cities.keys())
        bmu = get_closest_neuron(neurons, norm_cities[chosen_city])  # Best matching unit
        update_neuron_weights(bmu, norm_cities[chosen_city], neurons, learning_rate, int(round(neighborhood_radius)))

        if i != number_of_iterations-1:
            if decay_type == EXPONENTIAL_DECAY:
                neighborhood_radius *= math.exp(-1.0/exponential_decay_constant)
                learning_rate *= math.exp(-1.0/exponential_decay_constant/4)
            elif decay_type == LINEAR_DECAY:
                neighborhood_radius -= nr_linear_decay_constant
                learning_rate -= lr_linear_decay_constant

        ####Statistics###
        if i % plot_interval == 0:
            # plot_map(norm_cities, neurons, save=False)
            # tour_distance, tour = get_tour(norm_cities, neurons, norm_scale)
            print("\nIteration", i)
            # print("Distance: ", tour_distance)
            print("Neighborhood radius:", neighborhood_radius)
            print("Learning rate: ", learning_rate)


    final_distance, final_tour = get_tour(norm_cities, neurons, norm_scale)
    print("\nFinished", number_of_iterations, "iterations")
    print("Neighborhood radius:", neighborhood_radius)
    print("Learning rate: ", learning_rate)
    print("Final distance after", number_of_iterations, "iterations:", final_distance)
    # plot_map(norm_cities, final_tour)

# Western Sahara - 29
# tsp_som(DATA_WESTERN_SAHARA, 5000, decay_type=STATIC_DECAY, plot_interval=1000, number_of_neurons=50, neighborhood_radius_fraction=0.1, learning_rate=0.1)
# tsp_som(DATA_WESTERN_SAHARA, 5000, decay_type=LINEAR_DECAY, plot_interval=1000, number_of_neurons=50, neighborhood_radius_fraction=0.1, learning_rate=0.1)
# tsp_som(DATA_WESTERN_SAHARA, 5000, decay_type=EXPONENTIAL_DECAY, plot_interval=1000, number_of_neurons=50, neighborhood_radius_fraction=0.1, learning_rate=0.1)

# DJIBOUTI - 38
# tsp_som(DATA_DJIBOUTI, 5000, decay_type=STATIC_DECAY, plot_interval=1000, number_of_neurons=60, neighborhood_radius_fraction=0.1, learning_rate=0.2)
# tsp_som(DATA_DJIBOUTI, 5000, decay_type=LINEAR_DECAY, plot_interval=10000, number_of_neurons=60, neighborhood_radius_fraction=0.1, learning_rate=0.1)
# tsp_som(DATA_DJIBOUTI, 5000, decay_type=EXPONENTIAL_DECAY, plot_interval=1000, number_of_neurons=60, neighborhood_radius_fraction=0.1, learning_rate=0.1)

# QATAR - 194
# tsp_som(DATA_QATAR, 5000, decay_type=STATIC_DECAY, plot_interval=10000, number_of_neurons=194, neighborhood_radius_fraction=0.1, learning_rate=0.1)
# tsp_som(DATA_QATAR, 5000, decay_type=LINEAR_DECAY, plot_interval=10000, number_of_neurons=194, neighborhood_radius_fraction=0.1, learning_rate=0.1)
# tsp_som(DATA_QATAR, 5000, decay_type=EXPONENTIAL_DECAY, plot_interval=10000, number_of_neurons=300, neighborhood_radius_fraction=0.1, learning_rate=0.1)

# Uruguay - 734
# tsp_som(DATA_URUGUAY, 10000, decay_type=STATIC_DECAY, plot_interval=10000, number_of_neurons=1000, neighborhood_radius_fraction=0.1, learning_rate=0.1)
tsp_som(DATA_URUGUAY, 5000, decay_type=LINEAR_DECAY, plot_interval=10000, number_of_neurons=1000, neighborhood_radius_fraction=0.1, learning_rate=0.3)
# tsp_som(DATA_URUGUAY, 5000, decay_type=EXPONENTIAL_DECAY, plot_interval=10000, number_of_neurons=1000, neighborhood_radius_fraction=0.1, learning_rate=0.3)

# city_data = normalize_coordinates(parse(DATA_URUGUAY))
#
# plot(city_data)



# print(exponential_decay(3, 1000/math.log(3)))