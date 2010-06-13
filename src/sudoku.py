# Sean Davis, Matthew Henderson, Andrew Smith (Berea) 4.1.2010

from __future__ import print_function

import random, itertools, string, operator, copy

import constraint, networkx, sympy, glpk

####################################################################
# Basic parameters
####################################################################

def n_rows(boxsize): return boxsize**2
def n_cols(boxsize): return boxsize**2
def n_boxes(boxsize): return boxsize**2
def n_symbols(boxsize): return max(n_rows(boxsize), n_cols(boxsize))
def n_cells(boxsize): return n_rows(boxsize)*n_cols(boxsize)

####################################################################
# Cell label functions
####################################################################

def cell(row, column, boxsize): return (row - 1) * n_rows(boxsize) + column
def column(cell, boxsize): return (cell - 1) % n_rows(boxsize) + 1
def row(cell, boxsize): return (cell - 1) / n_cols(boxsize) + 1

def box_representative(box, boxsize):
    i = boxsize * ((box - 1) / boxsize)
    j = boxsize * ((box - 1) % boxsize) + 1
    return boxsize**2*i + j

####################################################################
# Convenient ranges
####################################################################

def cells(boxsize): return range(1, n_cells(boxsize) + 1)
def symbols(boxsize): return range(1, n_symbols(boxsize) + 1)
def rows(boxsize): return range(1, n_rows(boxsize) + 1)
def cols(boxsize): return range(1, n_cols(boxsize) + 1)
def boxes(boxsize): return range(1, n_boxes(boxsize) + 1)

def row_r(row, boxsize):
    """Cell labels in 'row' of Sudoku puzzle of dimension 'boxsize'."""
    nr = n_rows(boxsize)
    return range(nr * (row - 1) + 1, nr * row + 1)

def col_r(column, boxsize):
    """Cell labels in 'column' of Sudoku puzzle of dimension 'boxsize'."""
    nc = n_cols(boxsize)
    ncl = n_cells(boxsize)
    return range(column, ncl + 1 - (nc - column), nc)

def box_r(box, boxsize):
    """Cell labels in 'box' of Sudoku puzzle of dimension 'boxsize'."""
    return [box_representative(box, boxsize) + j + k - 1 for j in range(0, boxsize * n_rows(boxsize), n_cols(boxsize)) for k in range(1, boxsize + 1)]

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
    return [box_r(box, boxsize) for box in boxes(boxsize)]

def puzzle_rows(puzzle, boxsize):
    """Cell values, ordered by row."""
    return [map(puzzle.get, row_r(row, boxsize)) for row in rows(boxsize)]

def puzzle_columns(puzzle, boxsize):
    """Cell values, ordered by column."""
    return [map(puzzle.get, col_r(column, boxsize)) for column in cols(boxsize)]

def puzzle_boxes(puzzle, boxsize):
    """Cell values, ordered by box."""
    return [map(puzzle.get, box_r(box, boxsize)) for box in boxes(boxsize)]

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
    """Convert an integer to a printable character."""
    return string.printable[i]

def printable_to_int(c):
    """Convert a printable character to a integer."""
    return string.printable.index(c)

def are_all_different(l):
    """Test whether all elements in range 'l' are different."""
    return all(itertools.starmap(operator.ne, ordered_pairs(l)))

def are_all_different_nested(l):
    """Test whether every range in range 'l' is a range of all different
    elements."""
    return all(map(are_all_different, l))

####################################################################
# Cell dependencies
####################################################################

def dependent_cells(boxsize):
    """List of all pairs (x, y) with x < y such that x and y either lie in the 
    same row, same column or same box."""
    return list(set(flatten(map(list, map(ordered_pairs, cells_by_row(boxsize) + cells_by_col(boxsize) + cells_by_box(boxsize))))))

####################################################################
# String/dictionary conversions
####################################################################

def strip_nl(puzzle_string):
    """Remove newline characters from a string."""
    return puzzle_string.replace('\n', '')

def dict_to_string(fixed, boxsize):
    """Returns a puzzle string of dimension 'boxsize' from a dictionary of 
    'fixed' cells."""
    s = ''
    for cell in cells(boxsize):
        symbol = fixed.get(cell)
        if symbol:
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
            d[cell] = int(printable_to_int(puzzle[cell - 1]))
    return d

def graph_to_dict(graph):
    """Colored graph to dictionary conversion."""
    nodes = graph.node
    return dict([(vertex, nodes[vertex].get('color')) for vertex in nodes])

####################################################################
# Puzzle printing
####################################################################

def print_puzzle_s(puzzle_string, boxsize, file = None):
    """Pretty printing of Sudoku puzzle strings."""
    nc = n_cols(boxsize)
    for row in rows(boxsize):
        print(puzzle_string[row * nc:(row + 1) * nc].replace('', ' '), file = file)

def print_puzzle_d(puzzle_d, boxsize, width = 2, rowend = "\n", file = None):
    """Pretty printing of Sudoku puzzle dictionaries."""
    format_string = '%' + str(width) + 'i'
    for row in rows(boxsize):
        for col in cols(boxsize):
            symbol = puzzle_d.get(cell(row, col, boxsize))
            if symbol:
                print(format_string % symbol, end = "", file = file)
            else:
                print((width - 1)*' ' + '.', end = "", file = file)
        print(end = rowend, file = file)

def print_puzzle(puzzle_d, boxsize, padding = 1, rowend = "\n", file = None):
    """Pretty printing of Sudoku puzzle dictionaries, using printable
    characters."""
    for row in rows(boxsize):
        for col in cols(boxsize):
            symbol = puzzle_d.get(cell(row, col, boxsize))
            if symbol:
                print(" "*padding + int_to_printable(symbol) + " "*padding, end="", file = file)
            else:
                print(' '*padding + '.' + ' '*padding, end = "", file = file)                 
        print(end = rowend, file = file)

def print_puzzles(puzzles, boxsize, padding = 0, rowend = "", puzzleend = "", file = None):
    for puzzle in puzzles:
        print_puzzle(puzzle, boxsize, padding, rowend, file)
        print(puzzleend, file = file)

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
        problem.addConstraint(constraint.AllDifferentConstraint(), row)

def add_col_constraints(problem, boxsize):
    """add_col_constraints(problem, boxsize)

    Adds to constraint problem 'problem', all_different constraints on columns."""
    for col in cells_by_col(boxsize):    
        problem.addConstraint(constraint.AllDifferentConstraint(), col)

def add_box_constraints(problem, boxsize):
    """add_box_constraints(problem, boxsize)

    Adds to constraint problem 'problem', all_different constraints on boxes."""
    for box in cells_by_box(boxsize):
        problem.addConstraint(constraint.AllDifferentConstraint(), box)

def empty_puzzle_as_CP(boxsize):
    """empty_puzzle(boxsize) -> constraint.Problem

    Returns a constraint problem representing an empty Sudoku puzzle of 
    box-dimension 'boxsize'."""
    p = constraint.Problem()
    p.addVariables(cells(boxsize), symbols(boxsize)) 
    add_row_constraints(p, boxsize)
    add_col_constraints(p, boxsize)
    add_box_constraints(p, boxsize)
    return p

def puzzle_as_CP(fixed, boxsize):
    """puzzle_as_CP(fixed, boxsize) -> constraint.Problem

    Returns a constraint problem representing a Sudoku puzzle, based on 
    'fixed' cell dictionary."""
    p = empty_puzzle_as_CP(boxsize)
    for cell in fixed:
        p.addConstraint(constraint.ExactSumConstraint(fixed[cell]), [cell])
    return p

####################################################################
# Graph models
####################################################################

def empty_puzzle_as_graph(boxsize):
    """empty_puzzle_as_graph(boxsize) -> networkx.Graph

    Returns the Sudoku graph of dimension 'boxsize'."""   
    g = networkx.Graph()
    g.add_nodes_from(cells(boxsize))
    g.add_edges_from(dependent_cells(boxsize))
    return g

def puzzle_as_graph(fixed, boxsize):
    """Graph model of Sudoku puzzle of dimension 'boxsize' with 'fixed'
    cells."""
    g = empty_puzzle_as_graph(boxsize)
    for cell in fixed:
        g.node[cell]['color'] = fixed[cell]
    return g

####################################################################
# Vertex coloring algorithms
####################################################################

def neighboring_colors(graph, node):
    """Returns list of colors used on neighbors of 'node' in 'graph'."""
    return filter(None, [graph.node[neighbor].get('color') for neighbor in graph.neighbors(node)])

def n_colors(graph):
    """The number of distinct colors used on vertices of 'graph'."""
    return len(set([graph.node[i]['color'] for i in graph.nodes()]))

def least_missing(colors):
    """The smallest integer not in 'colors'."""
    colors.sort()
    for color in colors:
        if color + 1 not in colors:
            return color + 1

def first_available_color(graph, node):
    """The first color not used on neighbors of 'node' in 'graph'."""
    used_colors = neighboring_colors(graph, node)
    if len(used_colors) == 0:
        return 1
    else:
        return least_missing(used_colors)

def saturation_degree(graph, node):
    """Saturation degree of 'node' in 'graph'."""
    return len(set(neighboring_colors(graph, node)))

class FirstAvailableColor():
    """First available color choice visitor."""

    def __call__(self, graph, node):
        return first_available_color(graph, node)

class InOrder():
    """Natural vertex ordering strategy."""

    def __init__(self, graph):
        self.graph = graph

    def __iter__(self):
        return self.graph.nodes_iter()

class RandomOrder():
    """Random vertex ordering strategy."""

    def __init__(self, graph):
        self.graph = graph
        self.nodes = self.graph.nodes()

    def __iter__(self):
        random.shuffle(self.nodes)
        return iter(self.nodes)

class DSATOrder():
    """Saturation degree vertex ordering strategy."""

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

def vertex_coloring(graph, nodes = InOrder, choose_color = FirstAvailableColor):
    """Generic vertex coloring algorithm. Node ordering specified by 'nodes'
    iterator. Color choice strategy specified by 'choose_color'."""
    nodes = nodes(graph)
    for node in nodes:
        if not graph.node[node].get('color'):
            graph.node[node]['color'] = choose_color()(graph, node)
    return graph

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
    return reduce(operator.mul, [(x - row) for row in rows(boxsize)])

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

def empty_puzzle_as_polynomial_system(boxsize):
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

def puzzle_as_polynomial_system(fixed, boxsize):
    """Polynomial system for Sudoku puzzle of dimension 'boxsize' with fixed
    cells given by 'fixed' dictionary."""
    return empty_puzzle_as_polynomial_system(boxsize) + fixed_cells_polynomials(fixed)

####################################################################
# Linear program models
####################################################################

def lp_matrix_ncols(boxsize): return n_cells(boxsize) * n_symbols(boxsize)

def lp_matrix_nrows(boxsize): return 4*boxsize**4 # what is the origin of this number?

def lp_vars(boxsize):
    """Variables for Sudoku puzzle linear program model."""
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
    return [lp_occ_eq(cells, symbol, boxsize) for cells in cells_r for symbol in symbols(boxsize)]

def lp_nonempty_eqs(boxsize):
    """Linear equations (as lists of coefficients) which correspond to 
    every cell having one symbol."""
    return [lp_nonempty_eq(cell, boxsize) for cell in cells(boxsize)]

def lp_coeffs(boxsize):
    """Linear equations (as lists of coefficients) which correspond to 
    the empty Sudoku puzzle."""
    return lp_occ_eqs(cells_by_row(boxsize), boxsize) + lp_occ_eqs(cells_by_col(boxsize), boxsize) + lp_occ_eqs(cells_by_box(boxsize), boxsize) + lp_nonempty_eqs(boxsize)

def empty_puzzle_as_lp(boxsize):
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

def puzzle_as_lp(fixed, boxsize):
    """Linear program for Sudoku with 'fixed' clues."""
    lp = empty_puzzle_as_lp(boxsize)
    for cell in fixed:
        symbol = fixed[cell]
        lp.rows.add(1)
        r = lp_matrix_ncols(boxsize)*[0]
        r[lp_col_index(cell, symbol, boxsize)] = 1
        lp.rows[-1].matrix = r
        lp.rows[-1].bounds = 1.0, 1.0
    return lp

def solve_lp_puzzle(lp, boxsize):
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
# Puzzle solving strategies
####################################################################

def solve_as_CP(fixed, boxsize):
    """Use constraint programming to solve Sudoku puzzle of dimension 'boxsize'
    with 'fixed' cells."""
    return puzzle_as_CP(fixed, boxsize).getSolution()

def solve_as_lp(fixed, boxsize):
    """Use linear programming to solve Sudoku puzzle of dimension 'boxsize'
    with 'fixed' cells."""
    return solve_lp_puzzle(puzzle_as_lp(fixed, boxsize), boxsize)

def solve_as_groebner(fixed, boxsize):
    """Use groebner bases algorithm to solve Sudoku puzzle of dimension 
    'boxsize' with 'fixed' cells."""
    g = puzzle_as_polynomial_system(fixed, boxsize)
    h = sympy.groebner(g, cell_symbols(boxsize), order='lex')
    s = sympy.solve(h, cell_symbols(boxsize))
    return s 

def solve_as_graph(fixed, boxsize):
    """Use vertex coloring to solve Sudoku puzzle of dimension 'boxsize'
    with 'fixed' cells."""
    g = puzzle_as_graph(fixed, boxsize)
    cg = vertex_coloring(g, DSATOrder)
    return graph_to_dict(cg)

def solve_puzzles(puzzles, boxsize, solve = solve_as_CP):
    """Solve every puzzle in iterable 'puzzles'."""
    return [solve(puzzle, boxsize) for puzzle in puzzles]

def solve_puzzles_s(puzzles_s, boxsize, solve = solve_as_CP):
    """Solve every puzzle string in iterable 'puzzles'."""
    return solve_puzzles(map(lambda puzzle:string_to_dict(puzzle, boxsize), puzzles_s), boxsize, solve)

####################################################################
# File handling
####################################################################

def dimacs_file(graph, outfile):
    """Output to 'outfile' a graph in Dimacs format."""
    out = open(outfile, 'w')
    out.write(dimacs_string(graph))
    out.close()

####################################################################
# Puzzle generators
####################################################################

def random_puzzle_f(puzzle, n_fixed, boxsize):
    """Returns a puzzle dictionary of a random Sudoku puzzle of 'fixed' size
    based on the Sudoku 'puzzle' dictionary."""
    fixed = copy.deepcopy(puzzle)
    keys = fixed.keys()
    random.shuffle(keys)
    indices = keys[:len(keys) - n_fixed]
    for i in indices:
        del fixed[i]
    return fixed

def random_puzzle(n_fixed, boxsize, solve = solve_as_CP):
    """Random puzzle generator, based on solution of empty puzzle."""
    s = solve({}, boxsize)
    return random_puzzle_f(s, n_fixed, boxsize)

####################################################################
# Verification
####################################################################

def is_row_latin(puzzle, boxsize):
    """Test latin-ness of 'puzzle' rows."""
    return are_all_different_nested(puzzle_rows(puzzle, boxsize))

def is_column_latin(puzzle, boxsize):
    """Test latin-ness of 'puzzle' columns."""
    return are_all_different_nested(puzzle_columns(puzzle, boxsize))

def is_box_latin(puzzle, boxsize):
    """Test latin-ness of 'puzzle' boxes."""
    return are_all_different_nested(puzzle_boxes(puzzle, boxsize))

def is_sudoku(puzzle, boxsize):
    """Test whether 'puzzle' is a Sudoku puzzle of dimension 'boxsize'."""
    return is_row_latin(puzzle, boxsize) and is_column_latin(puzzle, boxsize) and is_box_latin(puzzle, boxsize)

def is_solution(fixed, puzzle, boxsize):
    """Test whether 'fixed' cells have same values in 'puzzle'."""
    return all(map(lambda x: fixed[x] == puzzle[x], fixed))

def is_sudoku_solution(fixed, puzzle, boxsize):
    """Test whether 'puzzle' is a solution of 'fixed'."""
    return is_sudoku(puzzle, boxsize) and is_solution(fixed, puzzle, boxsize)

def is_sudoku_solution_s(fixed_s, puzzle_s, boxsize):
    """Test whether 'puzzle_s' string is a solution of 'fixed_s' string."""
    fixed = string_to_dict(fixed_s, boxsize)
    puzzle = string_to_dict(puzzle_s, boxsize)
    return is_sudoku_solution(fixed, puzzle, boxsize)

def verify_solutions(puzzles, solutions, boxsize, verify_solution = is_sudoku_solution):
    """Test whether the iterable 'puzzles' has a solution in the 
    corresponding position of iterable 'solutions'."""
    return all(itertools.imap(lambda puzzle, solution: verify_solution(puzzle, solution, boxsize), puzzles, solutions))

