# Sean Davis, Matthew Henderson, Andrew Smith (Berea) 4.1.2010

from constraint import *
from math import sqrt, floor
from random import choice, randrange, seed

def convert_to_sage(number_string):
    """Conversion of Sudoku puzzle strings.
    
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
    """List of cell labels of top-left cell of each box. """
    return [cell(i,j,boxsize) for i in range(1,n_rows(boxsize),boxsize) for j in range(1,n_cols(boxsize),boxsize)]

def rows(boxsize):
    """Cell labels ordered by row."""
    nr = n_rows(boxsize)
    return [range(nr*(i-1)+1,nr*i+1) for i in range(1,nr+1)]

def cols(boxsize):
    """Cell labels ordered by column."""
    nc = n_cols(boxsize)
    ncl = n_cells(boxsize)
    return [range(i,ncl+1-(nc-i),nc) for i in range(1,nc+1)]

def boxes(boxsize):
    """Cell labels ordered by box."""
    nr = n_rows(boxsize)
    nc = n_cols(boxsize)
    return [[i+j+k for j in range(0,boxsize*nr,nc) for k in range(0,boxsize)] for i in top_left_cells(boxsize)]

def add_row_constraints(problem, boxsize):
    """Add to constraint problem 'problem', all_different constraints on rows."""
    for row in rows(boxsize):
        problem.addConstraint(AllDifferentConstraint(), row)

def add_col_constraints(problem, boxsize):
    """Add to constraint problem 'problem', all_different constraints on columns."""
    for col in cols(boxsize):    
        problem.addConstraint(AllDifferentConstraint(), col)

def add_box_constraints(problem, boxsize):
    """Add to constraint problem 'problem', all_different constraints on boxes."""
    for box in boxes(boxsize):
        problem.addConstraint(AllDifferentConstraint(), box)

def empty_puzzle(boxsize):
    """Create a constraint problem representing an empty Sudoku puzzle of box-dimension 'boxsize'."""
    p = Problem()
    p.addVariables(cells(boxsize), symbols(boxsize)) 
    add_row_constraints(p, boxsize)
    add_col_constraints(p, boxsize)
    add_box_constraints(p, boxsize)
    return p

def puzzle(boxsize, clues):
    """Create a constraint problem representing a Sudoku puzzle, where the fixed
       cells are specified by 'clues' dictionary."""
    p = empty_puzzle(boxsize)
    for x in range(1, len(clues)+1):
        if clues[x] != 0:
            p.addConstraint(ExactSumConstraint(clues[x]), [x])
    return p

def random_puzzle(boxsize, solution, fixed):
    """From 'solution' dictionary, generate a random Sudoku puzzle with 'fixed'
       number of cells."""
    indices = []
    for x in range(1, len(solution)+1):
        indices.append(x)
    for i in range(n_cells(boxsize) - fixed):
        c = choice(indices)
        solution[c] = 0
        indices.remove(c)
    return puzzle(boxsize, solution)
	
def dict_to_sudoku_string(solution):
    """Conversion from dictionary to puzzle string."""
    string = ""
    for x in range(1, len(solution)+1):
        string = string + str(solution[x])
    return string

def make_sudoku_constraint(number_string):
    """Create constraint problem from 'number_string' Sudoku puzzle string."""
    if sqrt(len(number_string)) != floor(sqrt(len(number_string))):
        print "Invalid size string"
        return False

    size_of_string = len(number_string)
    boxsize = int(sqrt(sqrt(size_of_string)))
    p = empty_puzzle(boxsize)

    for x in range(size_of_string):
        if number_string[x] != "0":
            p.addConstraint(ExactSumConstraint(int(number_string[x])), [x+1])
    return p
	
def list_to_string(list):
    """Implemented since the dancing links algorithm returns a list."""
    output = ""
    for i in range(len(list)):
        output += str(list[i])
    return output

def solve_from_file(infile, outfile):
    """Gets all solutions for all items provided in a file.  Uses the dancing links algorithm. (Sage Function)"""
    solutions = []
    input = open(infile, 'r')
    output = open(outfile, 'w')
    unsolved = input.readlines()
    for x in range(len(unsolved)):
        s = Sudoku(convert_to_sage(unsolved[x].rstrip()))
        solutions = list(s.dlx())
        for i in range(len(solutions)):
            output.write(list_to_string(solutions[i]) + "\n")

