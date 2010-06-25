#!/usr/bin/python

from __future__ import print_function

import sudoku, sys, progressbar, math, getopt

def usage():
    print("Usage: ")

def run(infilename, output, boxsize, model, verbose = False, verify = False, count = False):

    puzzles = open(infilename, 'r')
    outfile = open(output, 'a')
    puzzles_d = map(lambda puzzle:sudoku.string_to_dict(puzzle, boxsize), puzzles)

    solutions = []

    pbar = progressbar.ProgressBar()

    n_puzzles = 0

    for puzzle in puzzles_d:
        p = sudoku.Puzzle(puzzle, boxsize)        
        if count:
            s = sudoku.well_formed_solution(p)
        else:
            s = sudoku.solve(p, model)
        solutions.append(s)
        if verbose:
            n_puzzles += 1
            percentage = 1.0 * n_puzzles/len(puzzles_d) * 100
            int_percentage = int(math.ceil(percentage))
            pbar.render(int_percentage)
        if verify:
            if not sudoku.is_sudoku_solution(p, s):
                print("Non-solution")
                sys.exit(2)
    print(s, file = outfile)
    
    puzzles.close()
    outfile.close()


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvcwo:b:m:")
    except getopt.GetoptError, err:
        print(str(err))
        usage()
        sys.exit(2)
    output = None
    verbose = False
    verify = False
    count = False
    model = 'CP'
    boxsize = 3
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o == "-c":
            verify = True
        elif o == "-w":
            count = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-o", "--output"):
            output = a
        elif o in ("-b", "--boxsize"):
            boxsize = int(a)
        elif o in ("-m", "--model"):
            model = a
        else:
            assert False, "unhandled option"
    run(args[0], output, boxsize, model, verbose, verify, count)

if __name__ == "__main__":
    main()

