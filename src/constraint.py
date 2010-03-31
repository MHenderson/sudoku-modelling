import constraint

def printSolutions(problem):
    solution = problem.getSolutions()
    keys = solution[0].keys()
    for x in range(len(solution)):
        print str(solution[x][keys[0]]) + ", " + str(solution[x][keys[1]]) + ", " + str(solution[x][keys[2]])

def n_solutions(problem):
    solution = problem.getSolutions()
    return len(solution)
