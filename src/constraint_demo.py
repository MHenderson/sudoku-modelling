# Sean Davis, Andrew Smith (Berea) 3.31.2010

from constraint import *

def printSolutions(problem):
    solution = problem.getSolutions()
    keys = solution[0].keys()
    for x in range(len(solution)):
        print str(solution[x][keys[0]]) + ", " + str(solution[x][keys[1]]) + ", " + str(solution[x][keys[2]])

def n_solutions(problem):
    solution = problem.getSolutions()
    return len(solution)

def shidoku_test():
    shidoku = Problem()
    shidoku.addVariables(range(1,16 + 1),range(1,4 + 1))
    shidoku.addConstraint(AllDifferentConstraint(),[1,2,3,4])
    shidoku.addConstraint(AllDifferentConstraint(),[5,6,7,8])
    shidoku.addConstraint(AllDifferentConstraint(),[9,10,11,12])
    shidoku.addConstraint(AllDifferentConstraint(),[13,14,15,16])
    shidoku.addConstraint(AllDifferentConstraint(),[1,5,9,13])
    shidoku.addConstraint(AllDifferentConstraint(),[2,6,10,14])
    shidoku.addConstraint(AllDifferentConstraint(),[3,7,11,15])
    shidoku.addConstraint(AllDifferentConstraint(),[4,8,12,16])
    shidoku.addConstraint(AllDifferentConstraint(),[1,2,5,6])
    shidoku.addConstraint(AllDifferentConstraint(),[3,4,7,8])
    shidoku.addConstraint(AllDifferentConstraint(),[9,10,13,14])
    shidoku.addConstraint(AllDifferentConstraint(),[11,12,15,16])
    shidoku.addConstraint(ExactSumConstraint(1),[7])
    return shidoku

