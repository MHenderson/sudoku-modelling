import sudoku, timeit
import matplotlib.pyplot as plt

def solve(n_fixed, boxsize, model = 'CP'):
    fixed = sudoku.random_puzzle(n_fixed, boxsize)
    return sudoku.solve(fixed, boxsize, model)

def average(values):
    return sum(values, 0.0) / len(values)

if __name__ == "__main__":
    boxsize = 3
    fixed_lower_bound = 0
    fixed_upper_bound = sudoku.n_cells(boxsize)
    iterations = 1
    setup_string = "from __main__ import solve"
    n_fixed_range = range(fixed_upper_bound, fixed_lower_bound - 1, -1)
    timings = []
    for n_fixed in n_fixed_range:
        experiment_string = "solve(" + str(n_fixed) + "," + str(boxsize) + ")"        
        t = timeit.Timer(experiment_string, setup_string)    
        timings.append(average(t.repeat(repeat = iterations, number = 1)))
    plt.plot(n_fixed_range, timings)
    plt.show()
