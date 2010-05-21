import sudoku, sympy
from timeit import Timer

def solve(n_fixed, boxsize):
    p = sudoku.random_from_CP(n_fixed, boxsize)
    g = sudoku.polynomial_system(p, boxsize)
    h = sympy.groebner(g,sudoku.cell_symbols(boxsize), order='lex')
    s = sympy.solve(h, sudoku.cell_symbols(boxsize))
    r = [p,s]
    return r

if __name__ == "__main__":
    setup_string = """from __main__ import solve"""
    experiment_string = "print solve(5, 2)"
    t = Timer(experiment_string, setup_string)
    print t.timeit(number=1)

