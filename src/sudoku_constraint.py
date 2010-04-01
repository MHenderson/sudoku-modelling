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
    
    # These constraints must be generalized
    p.addConstraint(AllDifferentConstraint(),[1,2,3,4])
    p.addConstraint(AllDifferentConstraint(),[5,6,7,8])
    p.addConstraint(AllDifferentConstraint(),[9,10,11,12])
    p.addConstraint(AllDifferentConstraint(),[13,14,15,16])
    p.addConstraint(AllDifferentConstraint(),[1,5,9,13])
    p.addConstraint(AllDifferentConstraint(),[2,6,10,14])
    p.addConstraint(AllDifferentConstraint(),[3,7,11,15])
    p.addConstraint(AllDifferentConstraint(),[4,8,12,16])
    p.addConstraint(AllDifferentConstraint(),[1,2,5,6])
    p.addConstraint(AllDifferentConstraint(),[3,4,7,8])
    p.addConstraint(AllDifferentConstraint(),[9,10,13,14])
    p.addConstraint(AllDifferentConstraint(),[11,12,15,16])
    return p

# If we want a 9x9 square, the input should be 9.
def generate_blank_sudoku(size): 
    answer = ""
    for x in range(0, size*size):
        answer = answer + "0"
    return make_sudoku_constraint(answer)

# Test case
# c = make_sudoku_constraint("1234432131422410")
