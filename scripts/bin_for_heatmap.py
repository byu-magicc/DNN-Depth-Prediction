import numpy as np
import math

# This file adds the bin_for_heatmap function that divides up the
# plot area into bins for use in a heatmap.

def bin_for_heatmap(predicted, actual, lims=[100,100], num_cells=[100,100]):
    bins = np.zeros(num_cells)
    p_axis_factor = (lims[0]+0.0)/num_cells[0]
    a_axis_factor = (lims[1]+0.0)/num_cells[1]
    for p,a in zip(predicted, actual):
        #figure out which bin the point should go in
        p_axis = math.floor(p/p_axis_factor)
        a_axis = math.floor(a/a_axis_factor)
        
        # if it tries to go in a bin smaller than what we have, shove it up
        p_axis = p_axis if p_axis >= 0 else 0
        a_axis = a_axis if a_axis >= 0 else 0

        #if it tries to go in a bin larger than what we have, shove it down
        p_axis = p_axis if p_axis < num_cells[0] else num_cells[0]-1
        a_axis = a_axis if a_axis < num_cells[1] else num_cells[1]-1
        bins[p_axis, a_axis] += 1
    return bins

if __name__ == "__main__":
    #test one, equal cells
    lims = [2, 2]
    num_cells = [2,2]
    predicted = [0, -0.1, 0.1, -0.1, 1, 1.5, 1.5, 2, 1.9, 1.9, 0.5]
    actual = [0, 0, 0.9, 0.9, 1, 0.5, 1.5, 2, 1.9, 2.1, 1.5]

    correct = np.array([[4, 1],[1, 5]], dtype=int)
    test_result = bin_for_heatmap(predicted, actual, lims, num_cells)
    if np.array_equal(correct, test_result):
        print("Passed test 1")
    else:
        print("Failed test 1")
        print("Correct:")
        print(correct)
        print()
        print("Result:")
        print(test_result)
        if (correct.shape != test_result.shape):
            print("Incorrect final size")
        print()

    # test 2, unequal cells, all occupied, equal spacing
    lims = [2,3]
    num_cells = [2, 3]
    predicted = [0.5, 0.5, 0.5, 1.5, 1.5, 1.5]
    actual    = [0.5, 1.5, 2.5, 0.5, 1.5, 2.5]
    correct = np.array([[1, 1, 1],[1, 1, 1]])
    test_result = bin_for_heatmap(predicted, actual, lims, num_cells)
    if np.array_equal(correct, test_result):
        print("Passed test 2")
    else:
        print("Failed test 2")
        print("Correct:")
        print(correct)
        print()
        print("Result:")
        print(test_result)
        print()
    # test 3, unequal number of cells, unequal spacing
    lims = [2, 2]
    num_cells = [2, 1]
    predicted = [0, -0.1, 0.1, -0.1, 1, 1.5, 1.5, 2, 1.9, 1.9, 0.5]
    actual = [0, 0, 0.9, 0.9, 1, 0.5, 1.5, 2, 1.9, 2.1, 1.5]
    correct = np.array([[5],[6]])
    test_result = bin_for_heatmap(predicted, actual, lims, num_cells)
    if np.array_equal(correct, test_result):
        print("Passed test 3")
    else:
        print("Failed test 3")
        print("Correct:")
        print(correct)
        print()
        print("Result:")
        print(test_result)