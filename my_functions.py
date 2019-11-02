# importing the required module
import matplotlib.pyplot as plt
import xlrd
import numpy
import csv


def make_random_matrix(num_rows=3, num_columns=3):
    from random import seed
    from random import randint
    seed(1)
    matrix = numpy.zeros((num_rows, num_columns))
    for i in range(num_rows):
        for j in range(num_columns):
            matrix[i][j] = randint(0, 100)/100
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


def run_scenario(input_struct):
    scenarioid, loc, matrix, x1_0, maxtime, resultspath = \
        init_scenario_data(input_struct)

    run_lin_dynamics(scenarioid, matrix, x1_0, maxtime, resultspath)
    return


def init_scenario_data(input_struct):
    scenarioid = int(input_struct[0])
    loc = input_struct[1]
    random_matrix_size = int(input_struct[2])
    x1_0 = input_struct[3]
    maxtime = int(input_struct[4])
    resultspath = input_struct[5]
    if loc == -1:
        matrix = make_random_matrix(random_matrix_size, random_matrix_size)
    else:
        # read_matrix_from_loc(loc)
        matrix = numpy.zeros(random_matrix_size, random_matrix_size)

    return scenarioid, loc, matrix, x1_0, maxtime, resultspath


def run_lin_dynamics(scenarioid, matrix, x1_0, maxtime, resultspath):
    write_to_file(resultspath, 'resultsmatrix' + str(scenarioid) + '.txt', str(matrix))
    statevector = numpy.ones(len(matrix))
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
    line_str = line_str + '\n'
    file_name = 'results' + str(scenarioid) + '.csv'
    write_to_file(resultspath, file_name, line_str)
    return


def step_lin_dynamics(statevector, matrix):

    dxdt = numpy.zeros(len(statevector))
    for i in range(len(statevector)):
        for j in range(len(statevector)):
            dxdt[i] = (dxdt[i] + matrix[i][j] * statevector[i] * 1/statevector[j])
    dxdt = dxdt * 0.01

    updatedstatevector = numpy.add(statevector, dxdt)
    return updatedstatevector


def print_statevector(statevector, i, resultspath, scenarioid):
    file_name = 'results' + str(scenarioid) + '.csv'
    line_str = str(i/100) + ','
    for j in range(len(statevector)):
        line_str = line_str + str(statevector[j]) + ','
    line_str = line_str + '\n'
    append_to_file(resultspath, file_name, line_str)
    return


def plot_scenario(scenariodata):
    resultspath = scenariodata[5]
    scenarioid = scenariodata[0]
    #matrix_str = read_matrix_output(resultspath)
    csv_data = read_csv_results(resultspath, 'results' + str(scenarioid))
    time = read_column(csv_data, 0, 1)
    num_of_nodes = count_columns(csv_data) - 1
    #create_empty_fig()
    for i in range(1, num_of_nodes):
        xidata = read_column(csv_data, i, 1)
        add_data_to_fig(time, xidata, 'x' + str(i))
    complete_fig('time', 'x(t)', 'Network Dynamics 1')
    save_figure(resultspath)
    return


def read_matrix_ouput(resultspath):
    return


def read_csv_results(resultspath, filename):
    csv_file = open(resultspath + filename, "r")
    csv_data = csv.reader(csv_file)
    return csv_data

def count_columns(csv_data):
    ncol = len(next(csv_data))
    return ncol


def read_column(csv_data, col_num, start_row):
    col = numpy.zeros(csv_data.line_num - 1)
    i = 0
    for rows in csv_data:
        col[i] = rows[col_num]
        i = i + 1
    return col[start_row:-1]

def add_data_to_fig(x1, y1, textlabel):
    plt.plot(x1, y1, label=textlabel)
    return

def complete_fig(xlabel, ylabel, title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    return


def save_figure(resultspath):

    return
