# importing the required module
import matplotlib.pyplot as plt
import xlrd
import numpy


def make_random_matrix(num_rows=3, num_columns=3):
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


def append_to_file(file_path, file_name, line_str):
    f = open(file_path + file_name, 'a')
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


def read_input(input_file_path_str):
    wb = xlrd.open_workbook(input_file_path_str)
    data = wb.sheet_by_index(0)
    return data


def run_scenario(input_struct, i):
    scenarioid, loc, matrix, x1_0, maxtime, resultspath = \
        init_scenario_data(input_struct, i)

    run_lin_dynamics(scenarioid, matrix, x1_0, maxtime, resultspath)
    return


def init_scenario_data(input_struct, i):
    scenarioid = input_struct(i, 0)
    loc = input_struct(i, 1)
    random_matrix_size = input_struct(i, 2)
    x1_0 = input_struct(i, 3)
    maxtime = input_struct(i, 4)
    resultspath = input_struct(i, 5)
    if loc == -1:
        matrix = make_random_matrix(random_matrix_size, random_matrix_size)
    else:
        # read_matrix_from_loc(loc)
        matrix = numpy.zeros(random_matrix_size, random_matrix_size)

    return scenarioid, loc, matrix, x1_0, maxtime, resultspath


def run_lin_dynamics(scenarioid, matrix, x1_0, maxtime, resultspath):
    statevector = numpy.zeros(len(matrix), 1)
    statevector[0] = x1_0
    init_output_file(resultspath, scenarioid, statevector)
    for i in range(maxtime*100):
        print('Advancing dynamics step ' + str(i/100))
        statevector = step_lin_dynamics(statevector, matrix)
        print_statevector(statevector, i, resultspath, scenarioid)
    return


def init_output_file(resultspath, scenarioid, statevector):
    line_str = 'time,'
    for i in range(len(statevector)):
        line_str = line_str + 'x' + str(i+1) + ','
    line_str = line_str + '/n'
    file_name = 'results' + str(scenarioid) + '.csv'
    write_to_file(resultspath, file_name, line_str)
    return


def step_lin_dynamics(statevector, matrix):
    updatedstatevector = statevector
    dxdt = numpy.zeros(len(statevector), 1)
    for i in range(len(statevector)):
        for j in range(len(statevector)):
            dxdt[i] = dxdt[i] + matrix[i][j] * statevector[i] * statevector[j]
    updatedstatevector = numpy.add(statevector, dxdt)
    return updatedstatevector


def print_statevector(statevector, i, resultspath, scenarioid):
    file_name = 'results' + str(scenarioid) + '.csv'
    line_str = str(i/100) + ','
    for j in range(len(statevector)):
        line_str = line_str + str(statevector[i]) + ','
    line_str = line_str + '/n'
    append_to_file(resultspath, file_name, line_str)
    return


