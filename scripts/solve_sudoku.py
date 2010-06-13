#!/usr/bin/python

import sudoku, sys

puzzles = open(sys.argv[1], 'r')
solutions = open(sys.argv[2], 'a')
boxsize = int(sys.argv[3])
model = str(sys.argv[4])

s = sudoku.solve_puzzles_s(puzzles, boxsize, model)
if sudoku.verify_solutions(puzzles, s, boxsize, verify_solution = sudoku.is_sudoku_solution_s):
    sudoku.print_puzzles(s, boxsize, file = solutions)
else:
    pass

puzzles.close()
solutions.close()

