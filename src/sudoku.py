# Sean Davis, Matthew Henderson, Andrew Smith (Berea) 4.1.2010

from math import sqrt, floor
from random import choice, randrange, seed
import itertools
from copy import deepcopy

from constraint import Problem, AllDifferentConstraint, ExactSumConstraint
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
    """All ordered pairs from objects in 'range'."""
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
    """List of all pairs (x, y) with x < y such that x and y either lie in the 
    same row, same column or same box."""
    return list(set(flatten(map(list,map(ordered_pairs, rows(boxsize) + cols(boxsize) + boxes(boxsize))))))

####################################################################
# String handling
####################################################################

def convert_to_sage(number_string):
    """convert_to_sage(number_string) -> string

    Returns a converted sudoku puzzle string.
    After conversion an empty cell is represented by period instead of 0."""
    return number_string.replace('0','.')

def dict_to_sudoku_string(fixed, boxsize):
    """Returns a puzzle string of dimension 'boxsize' from a dictionary of 
    'fixed' cells."""
    s = ''
    for i in range(1, n_cells(boxsize) + 1):
        symbol = fixed.get(i)
        if symbol is not None:
            s += str(symbol)
        else:
            s += '.'
    return s

def sudoku_string_to_dict(puzzle):
    """Returns a dictionary based on a Sudoku puzzle string."""
    d = {}
    for i in range(len(puzzle)):
        if puzzle[i] != '.':
            d[i+1]=int(puzzle[i])
    return d

def print_puzzle(puzzle_string, boxsize):
    """Pretty printing of Sudoku puzzles."""
    nc = n_cols(boxsize)
    for row in range(n_rows(boxsize)):
        print puzzle_string[row*nc:(row + 1)*nc].replace('', ' ')

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
    for clue in clues:
        p.addConstraint(ExactSumConstraint(clues[clue]), [clue])
    return p

def make_sudoku_constraint(puzzle_string, boxsize):
    """make_sudoku_constraint(puzzle_string, boxsize) -> constraint.Problem

    Returns a constraint problem representing a Sudoku puzzle from the 
    'puzzle_string' Sudoku puzzle string.
    
    >>> p = "79....3.......69..8...3..76.....5..2..54187..4..7.....61..9...8..23.......9....54"
    >>> c = make_sudoku_constraint(p,3) 
    >>> s = dict_to_sudoku_string(c.getSolution(),3) """

    return puzzle(boxsize, sudoku_string_to_dict(puzzle_string))

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

def neighboring_colors(graph, node):
    """Returns list of colors used on neighbors of 'node' in 'graph'."""
    colors = []
    for node in graph.neighbors(node):
        color = graph.node[node].get('color')
        if color is not None:
            colors.append(color)
    return colors

def n_colors(graph):
    """The number of colors used on vertices of 'graph'."""
    return len(set([graph.node[i]['color'] for i in graph.nodes()]))

class FirstAvailableColorStrategy():

    def least_missing(self, colors):
        colors.sort()
        for color in colors:
            if color + 1 not in colors:
                return color + 1

    def first_available_color(self, graph, node):
        used_colors = neighboring_colors(graph, node)
        if len(used_colors) == 0:
            return 1
        else:
            return self.least_missing(used_colors)

    def __call__(self, graph, node):
        return self.first_available_color(graph, node)

def greedy_vertex_coloring(graph, nodes, choose_color = FirstAvailableColorStrategy()):
    """Color vertices sequentially, in order specified by 'nodes', according
    to given 'choose_color' strategy."""
    for node in nodes:
        graph.node[node]['color'] = choose_color(graph, node)
    return graph

def dimacs_string(graph):
    """Returns a string in Dimacs-format representing 'graph'."""
    s = ""
    s += "p " + "edge " + str(graph.order()) + " " + str(graph.size()) + "\n"
    for edge in graph.edges():
        s += "e " + str(edge[0]) + " " + str(edge[1]) + "\n"
    return s

####################################################################
# Polynomial system models
####################################################################

def cell_symbol_names(boxsize):
    """The names of symbols (e.g. cell 1 has name 'x1') used in the polynomial
    representation."""
    return map(lambda cell:'x' + str(cell), cells(boxsize))

def cell_symbols(boxsize):
    """The cells as symbols."""
    return map(sympy.Symbol, cell_symbol_names(boxsize))

def symbolize(pair):
    """Turn a pair of symbol names into a pair of symbols."""
    return (sympy.Symbol('x' + str(pair[0])),sympy.Symbol('x' + str(pair[1])))

def dependent_symbols(boxsize):
    """The list of pairs of dependent cells as symbol pairs."""
    return map(symbolize, dependent_cells(boxsize))

def node_polynomial(x, boxsize):
    """The polynomial representing a cell corresponding to symbol 'x'."""
    return reduce(lambda x,y: x*y, [(x - i) for i in range(1, n_rows(boxsize) + 1)])

def edge_polynomial(x, y, boxsize):
    """The polynomials representing the dependency of cells corresponding to
    symbols 'x' and 'y'."""
    return sympy.expand(sympy.cancel((node_polynomial(x, boxsize) - node_polynomial(y, boxsize))/(x - y)))

def node_polynomials(boxsize):
    """All cell polynomials."""
    return [node_polynomial(x, boxsize) for x in cell_symbols(boxsize)]

def edge_polynomials(boxsize):
    """All dependency polynomials."""
    return [edge_polynomial(x, y, boxsize) for x, y in dependent_symbols(boxsize)]

def polynomial_system_empty(boxsize):
    """The polynomial system for an empty Sudoku puzzle of dimension 
    'boxsize'."""
    return node_polynomials(boxsize) + edge_polynomials(boxsize)

def fixed_cell_polynomial(cell, symbol):
    """A polynomial representing the assignment of symbol 'symbol' to the cell
    'cell'."""
    return sympy.Symbol('x' + str(cell)) - symbol

def fixed_cells_polynomials(fixed):
    """Polynomials representing assignments of symbols to cells given by
    'fixed' dictionary."""
    return [fixed_cell_polynomial(cell, symbol) for cell, symbol in fixed.iteritems()]

def polynomial_system(fixed, boxsize):
    """Polynomial system for Sudoku puzzle of dimension 'boxsize' with fixed
    cells given by 'fixed' dictionary.

    >>> fixed = {1:1, 2:2, 3:3, 4:4,
    ...          5:3, 6:4, 7:1, 8:2,
    ...          9:2, 10:1,11:4,12:3,
    ...          13:4,14:3,15:2}
    >>> p = polynomial_system(fixed, 2)
    >>> import sympy
    >>> g = sympy.groebner(p,cell_symbols(2),order='lex') """
    return polynomial_system_empty(boxsize) + fixed_cells_polynomials(fixed)

####################################################################
# Puzzle processing strategies
####################################################################

def process_puzzle(puzzle, boxsize):
    """process_puzzle(puzzle, boxsize) -> string

    Constraint processing strategy."""
    p = make_sudoku_constraint(puzzle, boxsize)
    return dict_to_sudoku_string(p.getSolution())

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

def dimacs_file(boxsize, outfile):
    """Output to 'outfile' an empty Sudoku graph of dimension 'boxsize'."""
    out = open(outfile, 'w')
    sg = empty_sudoku_graph(boxsize)
    out.write(dimacs_string(sg))

####################################################################
# Puzzle generators
####################################################################

def random_puzzle(puzzle, n_fixed, boxsize):
    """Returns a puzzle dictionary of a random Sudoku puzzle of 'fixed' size
    based on the Sudoku 'puzzle' dictionary."""
    fixed = deepcopy(puzzle)
    ncl = n_cells(boxsize)
    indices = range(1, ncl + 1)
    for i in range(ncl - n_fixed):
        c = choice(indices)
        del fixed[c]
        indices.remove(c)
    return fixed

def random_from_CP(n_fixed, boxsize):
    p = empty_puzzle(boxsize)
    s = p.getSolution()
    return random_puzzle(s, n_fixed, boxsize)

####################################################################
# Main entry point
####################################################################

if __name__ == "__main__":
    import doctest
    doctest.testmod()

