import numpy as np

def closest_point_index(arr, value):
	return (np.abs(arr - value)).argmin()


def closest_2_points_index(arr, value):
	two_indexes = (np.abs(arr - value)).argsort()[:2]
	return np.sort(two_indexes)


def print_and_write(text, file):
	print(text)
	file.write(text + "\n")