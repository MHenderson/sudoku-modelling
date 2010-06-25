from timeit import Timer

if __name__ == "__main__":
    setup_string = "from sudoku import Puzzle, count_solutions"
    experiment_string = """print count_solutions(Puzzle({},2))"""
    t = Timer(experiment_string, setup_string)
    print t.timeit(number=1)

