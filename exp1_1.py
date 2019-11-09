from typing import TextIO
# importing the required module
import matplotlib.pyplot as plt
from my_functions import * #make_random_matrix, write_to_file, read_whole_file, simple_plot, read_input

def exp1_1_main(input_file_path_str, recalc_flag, redraw_flag, first_scenario, last_scenario):
    input_struct = read_input(input_file_path_str)

    if recalc_flag:
        for i in range(first_scenario, last_scenario+1):
            print('Running scenario ' + str(i))
            run_scenario(input_struct.row_values(i))
            print('Finished running scenario ' + str(i))

    if redraw_flag:
        for i in range(first_scenario, last_scenario+1):
            print('Plotting scenario  ' + str(i))
            plot_scenario(input_struct.row_values(i))
            print('Finished plotting scneario ' + str(i))

    return


input_file_path_str = 'C:\\Users\\Guy\\PycharmProjects\\Exp1Scenarios\\Exp1-Input.xlsx'
exp1_1_main(input_file_path_str, 1, 1, 10, 10)