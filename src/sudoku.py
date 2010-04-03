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

def add_row_constraints(problem, boxsize):
    for x in range(boxsize):
        problem.addConstraint(AllDifferentConstraint(), range(1+(boxsize*x), boxsize+1+(boxsize*x)))

def add_col_constraints(problem, boxsize):
    for x in range(boxsize):    
        domain = []        
        for y in range(boxsize):
            domain.append(x+1+(boxsize*y))
        problem.addConstraint(AllDifferentConstraint(), domain) # This needed to be in the 'for y' loop

def empty_sudoku(boxsize):
    p = Problem()
    p.addVariables(range(1,boxsize**2 + 1),range(1,boxsize + 1))    # Adjusted the range to be boxsize^2 instead of boxsize^4.
    add_row_constraints(p, boxsize)
    add_col_constraints(p, boxsize)
    # add_box_constraints(p, boxsize)
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

