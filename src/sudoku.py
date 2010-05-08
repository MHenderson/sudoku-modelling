# Sean Davis, Matthew Henderson, Andrew Smith (Berea) 4.1.2010

from constraint import *
from math import sqrt, floor
from random import choice, randrange, seed
import itertools
import networkx
import sympy

####################################################################
# Basic parameters
####################################################################

def n_rows(boxsize): return boxsize**2
def n_cols(boxsize): return boxsize**2
def n_boxes(boxsize): return boxsize**2
def n_cells(boxsize): return n_rows(boxsize)*n_cols(boxsize)
def cell(i,j,boxsize): return (i-1)*n_rows(boxsize) + j
def cells(boxsize): return range(1, n_cells(boxsize) + 1)
def symbols(boxsize): return range(1, n_rows(boxsize) + 1)

####################################################################
# Convenient functions
####################################################################

def ordered_pairs(range):
    return itertools.combinations(range, 2)

def flatten(list_of_lists):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(list_of_lists)

####################################################################
# Cell dependencies
####################################################################

def top_left_cells(boxsize): 
    """top_left_cells(boxsize) -> list

    Returns a list of cell labels of the top-left cell of each box."""
    return [cell(i,j,boxsize) for i in range(1,n_rows(boxsize),boxsize) for j in range(1,n_cols(boxsize),boxsize)]

def rows(boxsize):
    """rows(boxsize) -> list

    Returns a list of cell labels ordered by row for the given boxsize."""
    nr = n_rows(boxsize)
    return [range(nr*(i-1)+1,nr*i+1) for i in range(1,nr+1)]

def cols(boxsize):
    """cols(boxsize) -> list

    Returns a list of cell labels ordered by column for the given boxsize."""
    nc = n_cols(boxsize)
    ncl = n_cells(boxsize)
    return [range(i,ncl+1-(nc-i),nc) for i in range(1,nc+1)]

def boxes(boxsize):
    """boxes(boxsize) -> list

    Returns a list of cell labels ordered by box for the given boxsize."""
    nr = n_rows(boxsize)
    nc = n_cols(boxsize)
    return [[i+j+k for j in range(0,boxsize*nr,nc) for k in range(0,boxsize)] for i in top_left_cells(boxsize)]

def dependent_cells(boxsize):
    return list(flatten(map(list,map(ordered_pairs, rows(boxsize) + cols(boxsize) + boxes(boxsize)))))

####################################################################
# String handling
####################################################################

def convert_to_sage(number_string):
    """convert_to_sage(number_string) -> string

    Returns a converted sudoku puzzle string.
    After conversion an empty cell is represented by period instead of 0."""
    return number_string.replace('0','.')

def dict_to_sudoku_string(solution):
    """dict_to_sudoku_string(solution) -> string

    Returns a puzzle string converted from the 'solution' dictionary."""
    string = ""
    for x in range(1, len(solution)+1):
        string = string + str(solution[x])
    return string

####################################################################
# Constraint models
####################################################################

def add_row_constraints(problem, boxsize):
    """add_row_constraints(problem, boxsize)

    Adds to constraint problem 'problem', all_different constraints on rows."""
    for row in rows(boxsize):
        problem.addConstraint(AllDifferentConstraint(), row)

def add_col_constraints(problem, boxsize):
    """add_col_constraints(problem, boxsize)

    Adds to constraint problem 'problem', all_different constraints on columns."""
    for col in cols(boxsize):    
        problem.addConstraint(AllDifferentConstraint(), col)

def add_box_constraints(problem, boxsize):
    """add_box_constraints(problem, boxsize)

    Adds to constraint problem 'problem', all_different constraints on boxes."""
    for box in boxes(boxsize):
        problem.addConstraint(AllDifferentConstraint(), box)

def empty_puzzle(boxsize):
    """empty_puzzle(boxsize) -> constraint.Problem

    Returns a constraint problem representing an empty Sudoku puzzle of 
    box-dimension 'boxsize'."""
    p = Problem()
    p.addVariables(cells(boxsize), symbols(boxsize)) 
    add_row_constraints(p, boxsize)
    add_col_constraints(p, boxsize)
    add_box_constraints(p, boxsize)
    return p

def puzzle(boxsize, clues):
    """puzzle(boxsize, clues) -> constraint.Problem

    Returns a constraint problem representing a Sudoku puzzle, where the fixed
    cells are specified by 'clues' dictionary."""
    p = empty_puzzle(boxsize)
    for x in range(1, len(clues)+1):
        if clues[x] != 0:
            p.addConstraint(ExactSumConstraint(clues[x]), [x])
    return p

def random_puzzle(boxsize, solution, fixed):
    """random_puzzle(boxsize, solution, fixed) -> constraint.Problem

    Returns a constraint problem representing a random Sudoku puzzle of 'fixed' 
    size from the 'solution' dictionary."""
    indices = []
    for x in range(1, len(solution)+1):
        indices.append(x)
    for i in range(n_cells(boxsize) - fixed):
        c = choice(indices)
        solution[c] = 0
        indices.remove(c)
    return puzzle(boxsize, solution)
	
def make_sudoku_constraint(number_string, boxsize):
    """make_sudoku_constraint(number_string, boxsize) -> constraint.Problem

    Returns a constraint problem representing a Sudoku puzzle from the 
    'number_string' Sudoku puzzle string."""
    p = empty_puzzle(boxsize)
    for x in range(n_cells(boxsize)):
        if number_string[x] != "0":
            p.addConstraint(ExactSumConstraint(int(number_string[x])), [x+1])
    return p
	
def process_puzzle(puzzle, boxsize):
    """process_puzzle(puzzle, boxsize) -> string

    Returns a solved Sudoku puzzle string from the Sudoku string 'puzzle'.
    Constraint processing strategy."""
    p = make_sudoku_constraint(puzzle, boxsize)
    return dict_to_sudoku_string(p.getSolution())

####################################################################
# Graph models
####################################################################

def empty_sudoku_graph(boxsize):
    """empty_sudoku_graph(boxsize) -> networkx.Graph

    Returns the Sudoku graph of dimension 'boxsize'.
    
    >>> g = empty_sudoku_graph(3)
    >>> g = empty_sudoku_graph(4)"""
    
    g = networkx.Graph()
    g.add_nodes_from(cells(boxsize))
    g.add_edges_from(dependent_cells(boxsize))
    return g

####################################################################
# Polynomial system models
####################################################################

def cell_symbol_names(boxsize):
    return map(lambda cell:'x' + str(cell), cells(boxsize))

def cell_symbols(boxsize):
    return map(sympy.Symbol, symbol_names(boxsize))

def symbolize(pair):
    return (sympy.Symbol('x' + str(pair[0])),sympy.Symbol('x' + str(pair[1])))

def dependent_symbols(boxsize):
    return map(symbolize, dependent_cells(boxsize))

def node_polynomial(x, boxsize):
    return reduce(lambda x,y: x*y, [(x - i) for i in range(1, n_rows(boxsize) + 1)])

def edge_polynomial(x, y, boxsize):
    return sympy.cancel((node_polynomial(x, boxsize) - node_polynomial(y, boxsize))/(x - y))

def node_polynomials(boxsize):
    return [node_polynomial(x, boxsize) for x in cell_symbols(boxsize)]

def edge_polynomials(boxsize):
    return [edge_polynomial(x, y, boxsize) for (x,y) in dependent_symbols(boxsize)]

def polynomial_system(boxsize):
    return node_polynomials(boxsize) + edge_polynomials(boxsize)

####################################################################
# File handling
####################################################################

def solve_from_file(infile, outfile, boxsize):
    """solve_from_file(infile, outfile, boxsize)

    Outputs solutions to puzzles in file 'infile' to file 'outfile'."""
    input = open(infile, 'r')
    output = open(outfile, 'w')
    puzzles = input.readlines()
    for puzzle in puzzles:
        s = process_puzzle(puzzle, boxsize)
        output.write(s + "\n")

if __name__ == "__main__":
    import doctest
    doctest.testmod()

