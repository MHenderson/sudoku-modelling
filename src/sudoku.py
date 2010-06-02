# Sean Davis, Matthew Henderson, Andrew Smith (Berea) 4.1.2010

from __future__ import print_function

from math import sqrt, floor
from random import choice, seed, shuffle
import itertools, string
from copy import deepcopy

from constraint import Problem, AllDifferentConstraint, ExactSumConstraint
import networkx, sympy, glpk

####################################################################
# Basic parameters
####################################################################

def n_rows(boxsize): return boxsize**2
def n_cols(boxsize): return boxsize**2
def n_boxes(boxsize): return boxsize**2
def n_symbols(boxsize): return boxsize**2
def n_cells(boxsize): return n_rows(boxsize)*n_cols(boxsize)

####################################################################
# Cell label functions
####################################################################

def cell(row, column, boxsize): return (row - 1)*n_rows(boxsize) + column
def column(cell, boxsize): return (cell - 1) % n_rows(boxsize) + 1
def row(cell, boxsize): return (cell - 1) / n_cols(boxsize) + 1

def box_representatives(boxsize): 
    """box_representatives(boxsize) -> list

    Returns a list of cell labels of the top-left cell of each box."""
    return [cell(i, j, boxsize) for i in range(1, n_rows(boxsize), boxsize) for j in range(1, n_cols(boxsize), boxsize)]

####################################################################
# Convenient ranges
####################################################################

def cells(boxsize): return range(1, n_cells(boxsize) + 1)
def symbols(boxsize): return range(1, n_symbols(boxsize) + 1)
def rows(boxsize): return range(1, n_rows(boxsize) + 1)
def cols(boxsize): return range(1, n_cols(boxsize) + 1)

def row_r(row, boxsize):
    nr = n_rows(boxsize)
    return range(nr * (row - 1) + 1, nr * row + 1)

def col_r(column, boxsize):
    nc = n_cols(boxsize)
    ncl = n_cells(boxsize)
    return range(column, ncl + 1 - (nc - column), nc)

def box_r(box_representative, boxsize):
    nr = n_rows(boxsize)
    nc = n_cols(boxsize)
    return [box_representative + j + k - 1 for j in range(0, boxsize * nr, nc) for k in range(1, boxsize + 1)]

####################################################################
# Convenient functions
####################################################################

def ordered_pairs(range):
    """All ordered pairs from objects in 'range'."""
    return itertools.combinations(range, 2)

def flatten(list_of_lists):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(list_of_lists)

def int_to_printable(i):
    return string.printable[i]

####################################################################
# Cell dependencies
####################################################################

def cells_by_row(boxsize):
    """cells_by_row(boxsize) -> list

    Returns a list of cell labels ordered by row for the given boxsize."""
    return [row_r(row, boxsize) for row in rows(boxsize)]

def cells_by_col(boxsize):
    """cells_by_col(boxsize) -> list

    Returns a list of cell labels ordered by column for the given boxsize."""
    return [col_r(column, boxsize) for column in cols(boxsize)]

def cells_by_box(boxsize):
    """cells_by_box(boxsize) -> list

    Returns a list of cell labels ordered by box for the given boxsize."""
    return [box_r(box_representative, boxsize) for box_representative in box_representatives(boxsize)]

def dependent_cells(boxsize):
    """List of all pairs (x, y) with x < y such that x and y either lie in the 
    same row, same column or same box."""
    return list(set(flatten(map(list,map(ordered_pairs, cells_by_row(boxsize) + cells_by_col(boxsize) + cells_by_box(boxsize))))))

####################################################################
# String/Dictionary handling
####################################################################

def strip_nl(puzzle_string):
    return puzzle_string.replace('\n', '')

def dict_to_string(fixed, boxsize):
    """Returns a puzzle string of dimension 'boxsize' from a dictionary of 
    'fixed' cells."""
    s = ''
    for cell in cells(boxsize):
        symbol = fixed.get(cell)
        if symbol is not None:
            s += int_to_printable(symbol)
        else:
            s += '.'
    return s

def string_to_dict(puzzle, boxsize):
    """Returns a dictionary based on a Sudoku puzzle string."""
    puzzle = strip_nl(puzzle)
    d = {}
    for cell in cells(boxsize):
        if puzzle[cell - 1] != '.':
            d[cell] = int(puzzle[cell - 1])
    return d

def graph_to_dict(graph):
    """Colored graph to dictionary conversion."""
    nodes = graph.node
    return dict([(vertex, nodes[vertex].get('color')) for vertex in nodes])

####################################################################
# Puzzle printing
####################################################################

def print_puzzle(puzzle_string, boxsize, file = None):
    """Pretty printing of Sudoku puzzle strings."""
    nc = n_cols(boxsize)
    for row in rows(boxsize):
        print(puzzle_string[row * nc:(row + 1) * nc].replace('', ' '), file = file)

def print_puzzle_d(puzzle_d, boxsize, width = 2, rowend = "\n", file = None):
    """Pretty printing of Sudoku puzzle dictionaries."""
    fs = ''
    format_string = '%' + str(width) + 'i'
    for row in rows(boxsize):
        s = ''
        for col in cols(boxsize):
            symbol = puzzle_d.get(cell(row, col, boxsize))
            if symbol is not None:
                print(format_string % symbol, end = "", file = file)
            else:
                print((width - 1)*' ' + '.', end = "", file = file)
        print(end = rowend, file = file)

def print_puzzle_d_p(puzzle_d, boxsize, rowend = "\n", file = None):
    """Pretty printing of Sudoku puzzle dictionaries, using printable
    characters."""
    for row in rows(boxsize):
        s = ''
        for col in cols(boxsize):
            symbol = puzzle_d.get(cell(row, col, boxsize))
            if symbol is not None:
                print(int_to_printable(symbol), end="", file = file)
            else:
                print('.', end = "", file = file)                 
        print(end = rowend, file = file)

####################################################################
# Graph output
####################################################################

def dimacs_string(graph):
    """Returns a string in Dimacs-format representing 'graph'."""
    s = ""
    s += "p " + "edge " + str(graph.order()) + " " + str(graph.size()) + "\n"
    for edge in graph.edges():
        s += "e " + str(edge[0]) + " " + str(edge[1]) + "\n"
    return s

####################################################################
# Constraint models
####################################################################

def add_row_constraints(problem, boxsize):
    """add_row_constraints(problem, boxsize)

    Adds to constraint problem 'problem', all_different constraints on rows."""
    for row in cells_by_row(boxsize):
        problem.addConstraint(AllDifferentConstraint(), row)

def add_col_constraints(problem, boxsize):
    """add_col_constraints(problem, boxsize)

    Adds to constraint problem 'problem', all_different constraints on columns."""
    for col in cells_by_col(boxsize):    
        problem.addConstraint(AllDifferentConstraint(), col)

def add_box_constraints(problem, boxsize):
    """add_box_constraints(problem, boxsize)

    Adds to constraint problem 'problem', all_different constraints on boxes."""
    for box in cells_by_box(boxsize):
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

def puzzle_as_CP(fixed, boxsize):
    """puzzle_as_CP(fixed, boxsize) -> constraint.Problem

    Returns a constraint problem representing a Sudoku puzzle, based on 
    'fixed' cell dictionary."""
    p = empty_puzzle(boxsize)
    for cell in fixed:
        p.addConstraint(ExactSumConstraint(fixed[cell]), [cell])
    return p

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

def puzzle_graph(fixed, boxsize):
    g = empty_sudoku_graph(boxsize)
    for cell in fixed:
        g.node[cell]['color'] = fixed[cell]
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

def least_missing(colors):
    colors.sort()
    for color in colors:
        if color + 1 not in colors:
            return color + 1

def first_available_color(graph, node):
    used_colors = neighboring_colors(graph, node)
    if len(used_colors) == 0:
        return 1
    else:
        return least_missing(used_colors)

def saturation_degree(graph, node):
    return len(set(neighboring_colors(graph, node)))

class FirstAvailableColor():

    def __call__(self, graph, node):
        return first_available_color(graph, node)

class InOrder():

    def __init__(self, graph):
        self.graph = graph

    def __iter__(self):
        return self.graph.nodes_iter()

class RandomOrder():

    def __init__(self, graph):
        self.graph = graph
        self.nodes = self.graph.nodes()

    def __iter__(self):
        shuffle(self.nodes)
        return iter(self.nodes)

class DSATIter():

    def __init__(self, graph):
        self.graph = graph
        self.nodes = self.graph.nodes()
        self.value = 0

    def dsatur(self, node):
        return saturation_degree(self.graph, node)

    def next(self):
        self.value += 1
        if self.value > self.graph.order(): raise StopIteration
        self.nodes.sort(key = self.dsatur)
        return self.nodes.pop()

    def __iter__(self):
        return self

class DSATOrder():

    def __init__(self, graph):
        self.graph = graph

    def __iter__(self):
        return DSATIter(self.graph)

def vertex_coloring(graph, nodes = InOrder, choose_color = FirstAvailableColor):
    nodes = nodes(graph)
    for node in nodes:
        if graph.node[node].get('color') is None:
            graph.node[node]['color'] = choose_color()(graph, node)
    return graph

def greedy_vertex_coloring(graph):
    """Color vertices sequentially, using first available color."""
    return vertex_coloring(graph)

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
# Linear program models
####################################################################

def lp_matrix_ncols(boxsize): return n_cells(boxsize) * n_symbols(boxsize)

def lp_matrix_nrows(boxsize): return 4*boxsize**4 # what is the origin of this number?

def lp_vars(boxsize):
    return list(itertools.product(cells(boxsize), symbols(boxsize)))

def lp_col_index(cell, symbol, boxsize):
    """The column of the coefficient matrix which corresponds to the variable
    representing the assignment of 'symbol' to 'cell'."""
    return (cell - 1)*n_symbols(boxsize) + symbol - 1

def lp_occ_eq(cells, symbol, boxsize):
    """Linear equation (as list of coefficients) which corresponds to the cells
    in 'cells' having one occurence of 'symbol'."""
    coeffs = lp_matrix_ncols(boxsize)*[0]
    for cell in cells:
        coeffs[lp_col_index(cell, symbol, boxsize)] = 1
    return coeffs

def lp_nonempty_eq(cell, boxsize):
    """Linear equation (as list of coefficients) which corresponds to 'cell' 
    being assigned a symbol from 'symbols'."""
    coeffs = lp_matrix_ncols(boxsize)*[0]
    for symbol in symbols(boxsize):
        coeffs[lp_col_index(cell, symbol, boxsize)] = 1
    return coeffs

def lp_occ_eqs(cells_r, boxsize):
    """Linear equations (as lists of coefficients) which correspond to the
    cells in cells_r having one occurence of every symbol."""
    eqns = []
    for cells in cells_r:
        for symbol in symbols(boxsize):
            eqns.append(lp_occ_eq(cells, symbol, boxsize))
    return eqns

def lp_nonempty_eqs(boxsize):
    """Linear equations (as lists of coefficients) which correspond to 
    every cell having one symbol."""
    eqns = []
    for cell in cells(boxsize):
        eqns.append(lp_nonempty_eq(cell, boxsize))
    return eqns

def lp_coeffs(boxsize):
    """Linear equations (as lists of coefficients) which correspond to 
    the empty Sudoku puzzle."""
    return lp_occ_eqs(cells_by_row(boxsize), boxsize) + lp_occ_eqs(cells_by_col(boxsize), boxsize) + lp_occ_eqs(cells_by_box(boxsize), boxsize) + lp_nonempty_eqs(boxsize)

def lp_empty(boxsize):
    """Linear program for empty Sudoku puzzle."""
    lp = glpk.LPX()
    lp.cols.add(lp_matrix_ncols(boxsize))
    lp.rows.add(lp_matrix_nrows(boxsize))
    for c in lp.cols:
        c.bounds = 0.0, 1.0
    for r in lp.rows:
        r.bounds = 1.0, 1.0
    lp.matrix = list(flatten(lp_coeffs(boxsize)))
    return lp

def lp_puzzle(fixed, boxsize):
    """Linear program for Sudoku with 'fixed' clues."""
    lp = lp_empty(boxsize)
    for cell in fixed:
        symbol = fixed[cell]
        lp.rows.add(1)
        r = lp_matrix_ncols(boxsize)*[0]
        r[lp_col_index(cell, symbol, boxsize)] = 1
        lp.rows[-1].matrix = r
        lp.rows[-1].bounds = 1.0, 1.0
    return lp

def solve_as_lp_(lp, boxsize):
    """Solve a linear program Sudoku and return puzzle dictionary."""
    lp.simplex()
    for col in lp.cols:
        col.kind = int
    lp.integer()
    names = lp_vars(boxsize)
    sol = {}
    for c in lp.cols:
        if c.value == 1:
            sol[names[c.index][0]] = names[c.index][1]
    return sol

####################################################################
# Puzzle solving and processing strategies
####################################################################

def process_puzzle(puzzle, boxsize):
    """process_puzzle(puzzle, boxsize) -> string

    Constraint processing strategy."""
    p = make_sudoku_constraint(puzzle, boxsize)
    return dict_to_string(p.getSolution())

def solve_as_CP(fixed, boxsize):
    return puzzle_as_CP(fixed, boxsize).getSolution()

def solve_as_lp(fixed, boxsize):
    return solve_as_lp_(lp_puzzle(fixed, boxsize), boxsize)

def solve_as_groebner(fixed, boxsize):
    g = polynomial_system(fixed, boxsize)
    h = sympy.groebner(g, cell_symbols(boxsize), order='lex')
    s = sympy.solve(h, cell_symbols(boxsize))
    return s 

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

