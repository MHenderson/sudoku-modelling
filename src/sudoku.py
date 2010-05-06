# Sean Davis, Matthew Henderson, Andrew Smith (Berea) 4.1.2010

from constraint import *
from math import sqrt, floor
from random import choice, randrange, seed
import itertools
import networkx
import sympy

def convert_to_sage(number_string):
    """convert_to_sage(number_string) -> string

    Returns a converted sudoku puzzle string.
    After conversion an empty cell is represented by period instead of 0."""
    return number_string.replace('0','.')

def n_rows(boxsize): return boxsize**2
def n_cols(boxsize): return boxsize**2
def n_boxes(boxsize): return boxsize**2
def n_cells(boxsize): return n_rows(boxsize)*n_cols(boxsize)
def cell(i,j,boxsize): return (i-1)*n_rows(boxsize) + j
def cells(boxsize): return range(1, n_cells(boxsize) + 1)
def symbols(boxsize): return range(1, n_rows(boxsize) + 1)

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
	
def dict_to_sudoku_string(solution):
    """dict_to_sudoku_string(solution) -> string

    Returns a puzzle string converted from the 'solution' dictionary."""
    string = ""
    for x in range(1, len(solution)+1):
        string = string + str(solution[x])
    return string

def make_sudoku_constraint(number_string, boxsize):
    """make_sudoku_constraint(number_string, boxsize) -> constraint.Problem

    Returns a constraint problem representing a Sudoku puzzle from the 
    'number_string' Sudoku puzzle string."""
    p = empty_puzzle(boxsize)
    for x in range(n_cells(boxsize)):
        if number_string[x] != "0":
            p.addConstraint(ExactSumConstraint(int(number_string[x])), [x+1])
    return p
	
def list_to_string(list):
    """list_to_string(list) -> string

    Returns the string representation of 'list'.
    Implemented since the dancing links algorithm returns a list."""
    output = ""
    for i in range(len(list)):
        output += str(list[i])
    return output

def process_puzzle(puzzle, boxsize):
    """process_puzzle(puzzle, boxsize) -> string

    Returns a solved Sudoku puzzle string from the Sudoku string 'puzzle'.
    Constraint processing strategy."""
    p = make_sudoku_constraint(puzzle, boxsize)
    return dict_to_sudoku_string(p.getSolution())

def solve_from_file(infile, outfile, boxsize):
    """solve_from_file(infile, outfile, boxsize)

    Outputs solutions to puzzles in file 'infile' to file 'outfile'."""
    input = open(infile, 'r')
    output = open(outfile, 'w')
    puzzles = input.readlines()
    for puzzle in puzzles:
        s = process_puzzle(puzzle, boxsize)
        output.write(s + "\n")

def add_all_edges(graph, vertices):
    """add_all_edges(graph, vertices)

    Adds all edges between nodes in 'vertices' to 'graph'."""
    graph.add_edges_from(itertools.combinations(vertices, 2))

def empty_sudoku_graph(boxsize):
    """empty_sudoku_graph(boxsize) -> networkx.Graph

    Returns the Sudoku graph of dimension 'boxsize'.
    
    >>> g = empty_sudoku_graph(3)
    >>> g = empty_sudoku_graph(4)"""
    
    g = networkx.Graph()
    g.add_nodes_from(cells(boxsize))
    for vertices in rows(boxsize) + cols(boxsize) + boxes(boxsize):
        add_all_edges(g, vertices)
    return g

def F(x, boxsize):
    limit = n_rows(boxsize)
    return reduce(lambda x,y: x*y, [(x - i) for i in range(1, limit + 1)])

def G(x, y, boxsize):
    return sympy.cancel((F(x, boxsize) - F(y, boxsize))/(x - y))

def adjacent_symbols(boxsize):
    result = []
    vertices = rows(boxsize) + cols(boxsize) + boxes(boxsize)
    symbols = [[sympy.Symbol('x' + str(i)) for i in j] for j in vertices]
    for symbol in symbols:
        result += itertools.combinations(symbol,2)
    return result

def polynomial_system(boxsize):
    symbol_names = ['x' + str(cell) for cell in cells(boxsize)]
    symbols = [sympy.Symbol(name) for name in symbol_names]
    node_polynomials = [F(x, boxsize) for x in symbols]
    edge_polynomials = [G(x, y, boxsize) for (x,y) in adjacent_symbols(boxsize)]
    return node_polynomials + edge_polynomials

if __name__ == "__main__":
    import doctest
    doctest.testmod()

