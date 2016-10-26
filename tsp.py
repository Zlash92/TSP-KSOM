from parser import parse
from matplotlib import pyplot as plt
import math


DATA_WESTERN_SAHARA = 'data/wi29.tsp'
DATA_DJIBOUTI = 'data/dj38.tsp'
DATA_QATAR = 'data/qa194.tsp'
DATA_URUGUAY = 'data/uy734.tsp'


class Node:

    def __init__(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos


def get_distance(city1, city2):
    """
    Given tuples of (x,y) coordinates, compute distance between cities
    c = sqrt(a^2 + b^2)
    """
    a = max(city1[0], city2[0]) - min(city1[0], city2[0])
    b = max(city1[1], city2[1]) - min(city1[1], city2[1])
    return math.sqrt(a ** 2 + b ** 2)


def normalize_coordinates(city_data):
    result = {}
    # min_x = city_data[min(city_data, key=lambda i: city_data[i][0])]
    # min_y = city_data[min(city_data, key=lambda i: city_data[i][1])]
    #
    # max_x = city_data[max(city_data, key=lambda i: city_data[i][0])]
    max_y = max(city_data, key=lambda i: city_data[i][1])

    print max_y
    #
    # for city in city_data:
    #     result[city] = ( (city_data[city][0] - min_x) / max_x, (city_data[city][1] - min_y) / max_y)

    return result


def plot(city_data):
    for city in city_data:
        plt.scatter(city_data[city][0], city_data[city][1])
    plt.show()


def tsp_som(cities_datafile, learning_rate=0.1, neighborhood_radius=2):
    city_positions = parse(cities_datafile)  # Dictionary containing tuples of (x,y) coords
    number_of_neurons = len(city_positions)

    # norma_city_positions = normalize_coordinates(city_positions)
    # plot(city_positions)
    # plot(norma_city_positions)

tsp_som(DATA_WESTERN_SAHARA)