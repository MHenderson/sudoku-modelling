import sudoku, timeit
import matplotlib.pyplot as plt

def random_from_CP(n_fixed, boxsize):
    p = sudoku.empty_puzzle(boxsize)
    s = p.getSolution()
    return sudoku.random_puzzle(s, n_fixed, boxsize)

def solve(n_fixed, boxsize):
    fixed = random_from_CP(n_fixed, boxsize)
    p = sudoku.puzzle(boxsize, fixed)
    s = p.getSolution()

def average(values):
    return sum(values, 0.0) / len(values)

if __name__ == "__main__":
    boxsize = 3
    setup_string = "from __main__ import solve"
    n_fixed_range = range(81,0,-1)
    timings = []
    for n_fixed in n_fixed_range:
        experiment_string = "solve(" + str(n_fixed) + "," + str(boxsize) + ")"        
        t = timeit.Timer(experiment_string, setup_string)    
        timings.append(average(t.repeat(repeat=10,number=1)))
    plt.plot(n_fixed_range, timings)
    plt.show()
