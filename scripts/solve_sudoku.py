#!/usr/bin/python

import sudoku, sys, progressbar, math, getopt

def usage():
    print "Usage: "

def run(infilename, output, boxsize, model, verbose = False):

    puzzles = open(infilename, 'r')
    solutions = open(output, 'a')
    puzzles_d = map(lambda puzzle:sudoku.string_to_dict(puzzle, boxsize), puzzles)

    s = []

    p = progressbar.ProgressBar()

    n_puzzles = 0

    for puzzle in puzzles_d:
        s.append(sudoku.solve(puzzle, boxsize, model))
        if verbose:
            n_puzzles += 1
            percentage = 1.0 * n_puzzles/len(puzzles_d) * 100
            int_percentage = int(math.ceil(percentage))
            p.render(int_percentage)

    if sudoku.verify_solutions(puzzles, s, boxsize, verify_solution = sudoku.is_sudoku_solution_s):
        sudoku.print_puzzles(s, boxsize, file = solutions)
    else:
        pass

    puzzles.close()
    solutions.close()


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvo:b:m:")
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)
    output = None
    verbose = False
    model = 'CP'
    boxsize = 3
    for o, a in opts:
        if o == "-v":
            verbose = True
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
    run(args[0], output, boxsize, model, verbose)

if __name__ == "__main__":
    main()

