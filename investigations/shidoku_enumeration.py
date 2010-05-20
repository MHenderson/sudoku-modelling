from timeit import Timer

if __name__ == "__main__":
    setup_string = "from sudoku import empty_puzzle"
    experiment_string = """\
p = empty_puzzle(2)
s = p.getSolutions()
print len(s)"""
    t = Timer(experiment_string, setup_string)
    print t.timeit(number=1)

