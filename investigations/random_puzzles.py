import sudoku, timeit
import matplotlib.pyplot as plt

def random_by_CP(n_fixed, boxsize):
    fixed = sudoku.random_from_CP(n_fixed, boxsize)
    return sudoku.solve_as_CP(fixed, boxsize)

def random_by_lp(n_fixed, boxsize):
    fixed = sudoku.random_from_CP(n_fixed, boxsize)
    return sudoku.solve_as_lp(fixed, boxsize)

def random_by_groebner(n_fixed, boxsize):
    fixed = sudoku.random_from_CP(n_fixed, boxsize)
    return sudoku.solve_as_groebner(fixed, boxsize)

def solve(n_fixed, boxsize):
    return random_by_CP(n_fixed, boxsize)

def average(values):
    return sum(values, 0.0) / len(values)

if __name__ == "__main__":
    boxsize = 3
    fixed_lower_bound = 0
    fixed_upper_bound = sudoku.n_cells(boxsize)
    iterations = 10
    setup_string = "from __main__ import solve"
    n_fixed_range = range(fixed_upper_bound, fixed_lower_bound - 1, -1)
    timings = []
    for n_fixed in n_fixed_range:
        experiment_string = "solve(" + str(n_fixed) + "," + str(boxsize) + ")"        
        t = timeit.Timer(experiment_string, setup_string)    
        timings.append(average(t.repeat(repeat = iterations, number = 1)))
    plt.plot(n_fixed_range, timings)
    plt.show()
