# Sean Davis, Andrew Smith (Berea) 4.1.2010

from constraint import *
from math import sqrt, floor

def convert_to_sage(number_string):
    answer = ""
    for x in range(len(number_string)):
        if number_string[x] == "0":
            answer = answer + "."
        else:
            answer = answer + number_string[x]
    return answer

def make_sudoku_constraint(number_string):
    if sqrt(len(number_string)) != floor(sqrt(len(number_string))):
        print "Invalid size string"
        return False

    p = Problem()
    size_of_string = len(number_string)
    possible_values = int(sqrt(size_of_string))
    p.addVariables(range(1,size_of_string + 1),range(1,possible_values + 1))

    for x in range(size_of_string):
        if number_string[x] != "0":
            p.addConstraint(ExactSumConstraint(int(number_string[x])), [x+1])
    
    for x in range(possible_values):
    # Row Constraints
        p.addConstraint(AllDifferentConstraint(), range(1+(possible_values*x), possible_values+1+(possible_values*x)))
    # Column Constraints
        domain = []
        for y in range(possible_values):
            domain.append(x+1+(possible_values*y))
        p.addConstraint(AllDifferentConstraint(), domain)

    # Square Constraints

    return p
