# importing the required module
import matplotlib.pyplot as plt
def make_random_matrix(num_rows=3, num_columns=3):
    import numpy
    from random import seed
    from random import randint
    seed(1)
    matrix = numpy.zeros((num_rows, num_columns))
    for i in range(num_rows):
        for j in range(num_columns):
            matrix[i][j] = randint(0, 100)
    print(matrix)
    return matrix


def write_to_file(file_path, file_name, line_str):
    f = open(file_path + file_name, 'w+')
    f.write(line_str)
    f.close()
    return


def read_whole_file(file_path, file_name):
    f = open(file_path + file_name, 'r')
    if f.mode == 'r':
        contents = f.read()
    else:
        contents = ""
    f.close()
    return contents

def simple_plot(x_values, y_values, x_label, y_label, title):
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    return

