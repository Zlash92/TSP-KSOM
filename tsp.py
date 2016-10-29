from __future__ import print_function
from parser import parse
from matplotlib import pyplot as plt
import math
import random

DATA_WESTERN_SAHARA = 'data/wi29.tsp'
DATA_DJIBOUTI = 'data/dj38.tsp'
DATA_QATAR = 'data/qa194.tsp'
DATA_URUGUAY = 'data/uy734.tsp'


class Neuron:

    def __init__(self, id, x_pos, y_pos):
        self.id = id
        self.x_pos = x_pos
        self.y_pos = y_pos

    def show_data(self):
        print("ID:", self.id, "X:", self.x_pos, "Y:", self.y_pos)


def get_distance(city1, city2):
    """
    Given tuples of (x,y) coordinates, compute distance between cities
    c = sqrt(a^2 + b^2)
    """
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

    return normalized_city_coordinates


def get_closest_neuron(neurons, city_position):
    """
    Find neuron closest to given city coordinates
    :param neurons: list of neurons
    :param city_position: (x,y) tuple
    :return: neuron and distance
    """

    distances = []  # Distance to each neuron

    for n in neurons:  # type: Neuron
        distances.append(get_distance(city_position, (n.x_pos, n.y_pos)))

    min_distance = min(distances)
    bmu = neurons[distances.index(min_distance)]
    return bmu, min_distance


def plot(city_data, neuron_list=[], save=False, filename=''):
    plt.clf()
    for city in city_data:
        plt.scatter(city_data[city][0], city_data[city][1])

    for neuron in neuron_list:
        plt.scatter(neuron.x_pos, neuron.y_pos, c='r')

    if save:
        plt.savefig(filename)
    else:
        plt.show()


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

def adapt_neuron_positions(neuron, learning_rate, neighborbood_radius):
    pass


def tsp_som(cities_datafile, number_of_iterations, learning_rate=0.1, neighborhood_radius=2):
    # Dictionary containing tuples of (x,y) coordinates
    # Key=numbering from 0 and up. Value = (x,y)
    cities = parse(cities_datafile)

    number_of_neurons = len(cities)
    norm_cities = normalize_coordinates(cities)  # Dictionary containing normalized (x,y) coordinates
    neurons = initialize_neurons(number_of_neurons, norm_cities)

    plot(norm_cities, neurons, save=False, filename='ws_normalized.png')

    for i in range(number_of_iterations):
        chosen_city = random.choice(norm_cities.keys())
        bmu = get_closest_neuron(neurons, cities[chosen_city])  # Best matching unit
        adapt_neuron_positions(bmu, learning_rate, neighborhood_radius)


tsp_som(DATA_WESTERN_SAHARA, 10)
# tsp_som(DATA_URUGUAY, 10)

# city_data = normalize_coordinates(parse(DATA_URUGUAY))
#
# plot(city_data)



#TEST get closest neuron
# cities = {}
# cities[0] = (1, 2)
# neurons = [Neuron(0, 10, 10), Neuron(1, 4, 7), Neuron(2, 5, 5)]
# closest, distance = get_closest_neuron(neurons, cities[0])
# print(closest.id, closest.x_pos, closest.y_pos, distance)
# plot(cities, neurons)
