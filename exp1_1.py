from typing import TextIO
# importing the required module
import matplotlib.pyplot as plt
# This is a test
from my_functions import * #make_random_matrix, write_to_file, read_whole_file, simple_plot, read_input
# import make_random_matrix
from my_functions2 import test1

# data = read_input('C:\\Users\\Guy\\PycharmProjects\\Exp1Scenarios\\Exp1-Input.xlsx')
#
# print(data.cell_value(0, 0))
#
# print('Helllo')
#
# x = [1, 2, 3]
#
# for y in x:
#     print(y)
#
# print(test1(1, 2))
# test1(3, 5)
# make_random_matrix(3, 3)
# path = 'C:\\Users\\Guy\\PycharmProjects\\Exp1'
# file = open(path + '\\testfile.csv', "w+")
#
# file.write("This, is, a, test\n")
# file.write("To, add, more, lines.")
#
# file.close()
#
# write_to_file('C:\\Users\\Guy\\PycharmProjects\\Exp1', '\\testfile2.csv', '-1,-2,-3,1,2,3\n5,6,7,8,9,10')
#
# my_file = read_whole_file('C:\\Users\\Guy\\PycharmProjects\\Exp1', '\\testfile2.csv')
#
# print(my_file)
# print(my_file)
#
# simple_plot([1, 2, 3], [4, 6, 5], 'x values', 'y values', 'graph title')

def exp1_1_main(input_file_path_str):
    input_struct = read_input(input_file_path_str)

    for i in range(1, input_struct.nrows):
        print('Running scenario ' + str(i))
        run_scenario(input_struct.row_values(i))
        print('Finished running scenario ' + str(i))

    for i in range(1, input_struct.nrows):
        print('Plotting scenario  ' + str(i))
        plot_scenario(input_struct.row_values(i))
        print('Finished plotting scneario ' + str(i))

    return


input_file_path_str = 'C:\\Users\\Guy\\PycharmProjects\\Exp1Scenarios\\Exp1-Input.xlsx'
exp1_1_main(input_file_path_str)