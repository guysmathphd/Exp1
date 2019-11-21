# importing the required module
import matplotlib.pyplot as plt
import xlrd
import numpy
import csv
import os
import math
from rk4 import *
from IPython.display import display, Markdown, Latex

def make_random_matrix(num_rows=3, num_columns=3, seed_index=1):
    from random import seed
    from random import randint
    seed(seed_index)
    matrix = numpy.zeros((num_rows, num_columns))
    for i in range(num_rows):
        for j in range(num_columns):
            matrix[i][j] = randint(0, 100)/100
    print(matrix)
    return matrix


def write_to_file(file_path, file_name, line_str):
    if not os.path.exists(file_path):
        os.mkdir(file_path)
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
    scenarioid, loc, matrix, x1_0, maxtime, resultspath, dynamicModel, integration_method = \
        init_scenario_data(input_struct)

    run_lin_dynamics(scenarioid, matrix, x1_0, maxtime, resultspath, dynamicModel, integration_method)
    return


def init_scenario_data(input_struct):
    scenarioid = int(input_struct[0])
    loc = input_struct[1]
    random_matrix_size = int(input_struct[2])
    x1_0 = input_struct[3]
    maxtime = int(input_struct[4])
    resultspath = input_struct[5]
    seed_index = input_struct[6]
    dynamic_model = int(input_struct[7])
    integration_method = int(input_struct[8])
    if loc == -1:
        matrix = make_random_matrix(random_matrix_size, random_matrix_size, seed_index)
    else:
        matrix = load_matrix_from_file(loc, random_matrix_size)
    return scenarioid, loc, matrix, x1_0, maxtime, resultspath, dynamic_model, integration_method


def run_lin_dynamics(scenarioid, matrix, x1_0, maxtime, resultspath, dynamicModel, integration_method):
    write_to_file(resultspath, 'resultsmatrix' + str(scenarioid) + '.txt', str(matrix))
    statevector = numpy.ones(len(matrix))
    statevector[0] = x1_0
    time = 0
    init_output_file(resultspath, scenarioid, statevector)
    print_statevector(statevector, time, resultspath, scenarioid)
    #for i in range(1, maxtime*100):
    while time <= maxtime:
        print('Advancing dynamics step ' + str(time))
        statevector, time = step_lin_dynamics(statevector, time, matrix, dynamicModel, integration_method)
        # if dynamicModel == 1:
        #     statevector = step_lin_dynamics1(statevector, matrix)
        # else:
        #     if dynamicModel == 2:
        #         statevector = step_lin_dynamics2(statevector, matrix)
        print_statevector(statevector, time, resultspath, scenarioid)
    return


def init_output_file(resultspath, scenarioid, statevector):
    line_str = 'time,'
    for i in range(len(statevector)):
        line_str = line_str + 'x' + str(i+1) + ','
    line_str = line_str + '\n'
    file_name = 'results' + str(scenarioid) + '.csv'
    write_to_file(resultspath, file_name, line_str)
    return

def step_lin_dynamics(statevector, time, matrix, dynamicModel, integration_method = 1, time_step = 0.01):
    f = select_dynamic_model(dynamicModel)
    im = select_integration_method(integration_method)
    updatedstatevector = im(time, statevector, matrix, f, time_step)
    time = time + time_step
    return updatedstatevector, time


def select_dynamic_model(dynamicModel):
    if dynamicModel == 1:
        return dynamicModel1
    if dynamicModel == 2:
        return dynamicModel2
    if dynamicModel == 3:
        return dynamicModel3
    if dynamicModel == 4:
        return dynamicModel4
    if dynamicModel == 5:
        return dynamicModel5


def dynamicModel1(time, statevector, matrix):
    dxdt = numpy.zeros(len(statevector))
    for i in range(len(statevector)):
        for j in range(len(statevector)):
            dxdt[i] = (dxdt[i] + matrix[i][j] * statevector[i] * 1/statevector[j])
            return dxdt


def dynamicModel2(time, statevector, matrix):
    dxdt = numpy.zeros(len(statevector))
    for i in range(len(statevector)):
        dxdt[i] = 1 - statevector[i]
        for j in range(len(statevector)):
            dxdt[i] = dxdt[i] - matrix[i][j] * statevector[i] * statevector[j]
            return dxdt


def dynamicModel3(time, statevector, matrix):
    dxdt = numpy.zeros(len(statevector))
    for i in range(len(statevector)):
        dxdt[i] = math.cos(time)
        return dxdt


def dynamicModel4(time, statevector, matrix):
    dxdt = 1
    return dxdt


def dynamicModel5(time, statevector, matrix):
    dxdt = numpy.matmul(matrix, statevector)
    return dxdt


def select_integration_method(integration_method):
    if integration_method == 1:
        return my_euler, 'Euler integration'
    if integration_method == 2:
        return my_rk4, 'Runge Kutta 4th Order'


def my_euler(time, statevector, matrix, f, time_step):
    dxdt = f(time, statevector, matrix)
    dxdt = dxdt * time_step
    updatedstatevector = numpy.add(statevector, dxdt)
    return updatedstatevector


def my_rk4(time, statevector, matrix, f, time_step):
    def f_A(time, statevecotr):
        return f(time, statevector, matrix)
    updatedstatevector = rk4(time, statevector, time_step, f_A)
    return updatedstatevector

def step_lin_dynamics1(statevector, matrix):

    dxdt = numpy.zeros(len(statevector))
    for i in range(len(statevector)):
        for j in range(len(statevector)):
            dxdt[i] = (dxdt[i] + matrix[i][j] * statevector[i] * 1/statevector[j])
    dxdt = dxdt * 0.01
    updatedstatevector = numpy.add(statevector, dxdt)
    return updatedstatevector


def step_lin_dynamics2(statevector, matrix):
    dxdt = numpy.zeros(len(statevector))
    for i in range(len(statevector)):
        dxdt[i] = 1 - statevector[i]
        for j in range(len(statevector)):
            dxdt[i] = dxdt[i] - matrix[i][j] * statevector[i] * statevector[j]
    dx = dxdt * 0.01
    updatedstatevector = numpy.add(statevector, dx)
    return updatedstatevector


def print_statevector(statevector, time, resultspath, scenarioid):
    file_name = 'results' + str(scenarioid) + '.csv'
    line_str = str(time) + ','
    for j in range(len(statevector)):
        line_str = line_str + str(statevector[j]) + ','
    line_str = line_str + '\n'
    append_to_file(resultspath, file_name, line_str)
    return


def plot_scenario(scenariodata):
    resultspath = scenariodata[5]
    scenarioid = int(scenariodata[0])
    x1_0 = scenariodata[3]
    dynamic_model = int(scenariodata[7])
    integration_method = int(scenariodata[8])
    compare_to = int(scenariodata[9])
    matrix_str = read_matrix_output(resultspath, scenarioid)
    csv_data = read_csv_results(resultspath, 'results' + str(scenarioid) + '.csv')
    num_of_nodes = count_columns(csv_data) - 1
    num_rows_data = count_rows(csv_data) - 1
    f, ax = plt.subplots()
    for i in range(num_of_nodes):
        csv_data = read_csv_results(resultspath, 'results' + str(scenarioid) + '.csv')
        if i == 0:  # time column
            time = read_column(csv_data, 0, 1, num_rows_data)
        else:
            xidata = read_column(csv_data, i, 1, num_rows_data)
            add_data_to_fig(ax, time, xidata, r'$x_' + str(i) + '$')
    if compare_to != -1:
        fRef, ref_str = select_reference(compare_to)
        add_reference_to_fig(ax, time, fRef, ref_str)
    _, intMethStr = select_integration_method(integration_method)
    complete_fig(ax, 'Time', r'$x_i(t)$', 'Network Dynamics Scenario ' + str(scenarioid), matrix_str, dynamic_model, x1_0, intMethStr)
    save_figure(resultspath, str(scenarioid), 'fig001')
    return

def select_reference(compare_to):
    if compare_to == 1:
        fRef = numpy.sin
        ref_str = 'x = sin(t)'
    if compare_to == 2:
        fRef = numpy.exp
        ref_str = r'$x = e^t$'
    if compare_to == 3:
        def fRef3(t):
            return numpy.exp(t) + numpy.exp(-t)
        fRef = fRef3
        ref_str = r'$x = e^t + e^{-t}'
    if compare_to == 4:
        def fRef4(t):
            return numpy.exp(t) - numpy.exp(-t)
        fRef = fRef4
        ref_str = r'$x = e^t - e^{-t}'
    return fRef, ref_str


def add_reference_to_fig(ax, time, fRef, ref_str):
    add_data_to_fig(ax, time, fRef(time), ref_str)
    return


def read_matrix_output(resultspath, scenarioid):
    contents = read_whole_file(resultspath, 'resultsmatrix' + str(scenarioid) + '.txt')
    return contents


def load_matrix_from_file(matrix_file_path, size):
    with open(matrix_file_path, 'r') as f:
        matrix_temp = [[float(num) for num in line.split(',')] for line in f]
    matrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = matrix_temp[i][j]
    return matrix


def read_csv_results(resultspath, filename):
    csv_file = open(resultspath + filename, "r")
    csv_data = csv.reader(csv_file)
    csv_file.seek(0)
    return csv_data


def count_columns(csv_data):
    ncol = len(next(csv_data))
    return ncol


def count_rows(csv_data):
    i = 0
    for row in csv_data:
        i = i + 1
    return i


def read_column(csv_data, col_num, start_row, end_row):
    col = numpy.zeros(end_row - start_row + 1)
    i = 0  # counter for elements of col
    j = 0  # counter for rows
    for rows in csv_data:
        if j < start_row:
            j = j + 1
            continue
        else:
            if j > end_row:
                break
            else:
                col[i] = rows[col_num]
                j = j + 1
                i = i + 1
    return col


def add_data_to_fig(ax, x1, y1, textlabel):
    ax.plot(x1, y1, label=textlabel)
    return


def complete_fig(ax, xlabel, ylabel, title, text, dynamic_model, x1_0, intMethStr):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.text(0.3, 0.6, 'A' + r'$_i$' + r'$_j$' + ' = ' + '\n' + text, transform=ax.transAxes)
    if dynamic_model == 1:
        plt.text(0.1, 0.4, r'$\.x_i$' + '(t) = ' + r'$\sum_{i,j} A_{ij}x_i x_j^{-1}$', transform=ax.transAxes)
    if dynamic_model == 2:
            plt.text(0.1, 0.4, r'$\dfrac{dx_i}{dt}$' + ' = ' + r'$1 - x_i - \sum_{j} A_{ij}x_i x_j$', transform=ax.transAxes)
    plt.text(0.1, 0.5, intMethStr, transform=ax.transAxes)
    if x1_0 != 1:
        plt.text(0.1, 0.9, r'$x_1(0) = $' + str(x1_0), transform=ax.transAxes)
        i_str = 'for i > 0'
    else:
        i_str = 'for all i'
    plt.text(0.1, 0.8, r'$x_i(0) = 1, $' + i_str, transform=ax.transAxes)
    #  plt.show()
    return


def save_figure(resultspath, scenarioid, figname):
    plt.savefig(resultspath + scenarioid + '_fig001.png')
    plt.savefig(resultspath + scenarioid + '_fig001.pdf')
    plt.show()
    return
