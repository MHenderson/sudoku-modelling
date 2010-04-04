# Sean Davis, Matthew Henderson, Andrew Smith (Berea) 4.1.2010

from constraint import *
from math import sqrt, floor
from random import randrange, seed

def convert_to_sage(number_string):
    answer = ""
    for x in range(len(number_string)):
        if number_string[x] == "0":
            answer = answer + "."
        else:
            answer = answer + number_string[x]
    return answer

def n_rows(boxsize): return boxsize**2
def n_cols(boxsize): return boxsize**2
def n_boxes(boxsize): return boxsize**2
def n_cells(boxsize): return n_rows(boxsize)*n_cols(boxsize)
def cell(i,j,boxsize): return (i-1)*n_rows(boxsize) + j
def cells(boxsize): return range(1, n_cells(boxsize) + 1)
def symbols(boxsize): return range(1, n_rows(boxsize) + 1)

def top_left_cells(boxsize): 
    return [cell(i,j,boxsize) for i in range(1,n_rows(boxsize),boxsize) for j in range(1,n_cols(boxsize),boxsize)]

def rows(boxsize):
    nr = n_rows(boxsize)
    return [range(nr*(i-1)+1,nr*i+1) for i in range(1,nr+1)]

def cols(boxsize):
    nc = n_cols(boxsize)
    ncl = n_cells(boxsize)
    return [range(i,ncl+1-(nc-i),nc) for i in range(1,nc+1)]

def boxes(boxsize):
    nr = n_rows(boxsize)
    nc = n_cols(boxsize)
    return [[i+j+k for j in range(0,boxsize*nr,nc) for k in range(0,boxsize)] for i in top_left_cells(boxsize)]

def add_row_constraints(problem, boxsize):
    for row in rows(boxsize):
        problem.addConstraint(AllDifferentConstraint(), row)

def add_col_constraints(problem, boxsize):
    for col in cols(boxsize):    
        problem.addConstraint(AllDifferentConstraint(), col)

def add_box_constraints(problem, boxsize):
    for box in boxes(boxsize):
        problem.addConstraint(AllDifferentConstraint(), box)

def empty_sudoku(boxsize):
    p = Problem()
    p.addVariables(cells(boxsize), symbols(boxsize)) 
    add_row_constraints(p, boxsize)
    add_col_constraints(p, boxsize)
    add_box_constraints(p, boxsize)
    return p
	
def constraintSolution_to_sudokuString(solution):
	# This function takes the result of getSolutionIter().next() and returns it in string format.
    string = ""
    for x in range(1, len(solution)+1):
        string = string + str(solution[x])
    return string

def make_sudoku_constraint(number_string):
    if sqrt(len(number_string)) != floor(sqrt(len(number_string))):
        print "Invalid size string"
        return False

    size_of_string = len(number_string)
    boxsize = int(sqrt(size_of_string))
    p = empty_sudoku(boxsize)

    for x in range(size_of_string):
        if number_string[x] != "0":
            p.addConstraint(ExactSumConstraint(int(number_string[x])), [x+1])
    return p

# If we want a 9x9 square, the input should be 3.
def generate_blank_sudoku(size): 
    answer = ""
    for x in range(0, pow(size,4)):
        answer = answer + "0"
    return make_sudoku_constraint(answer)

# If we want a 9x9 square, the input should be 3.
# Note that this will rarely produce solvable puzzles
def generate_random_sudoku(size):
    answer = ""
    for x in range(0, pow(size,4)):
        answer = answer + str(randrange(0,size*size))
    return make_sudoku_constraint(answer)

