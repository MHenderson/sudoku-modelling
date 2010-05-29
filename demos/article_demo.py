In [1]: import sudoku

In [2]: p = """
   ...: 25..3.9.1
   ...: .1...4...
   ...: 4.7...2.8
   ...: ..52.....
   ...: ....981..
   ...: .4...3...
   ...: ...36..72
   ...: .7......3
   ...: 9.3...6.4
   ...: """
In [3]: sudoku.string_to_dict(p)
Out[3]: 
{1: 2,
 2: 5,
 5: 3,
 7: 9,
 9: 1,
 11: 1,
 15: 4,
 19: 4,
 21: 7,
 25: 2,
 27: 8,
 30: 5,
 31: 2,
 41: 9,
 42: 8,
 43: 1,
 47: 4,
 51: 3,
 58: 3,
 59: 6,
 62: 7,
 63: 2,
 65: 7,
 72: 3,
 73: 9,
 75: 3,
 79: 6,
 81: 4}

In [4]: c = sudoku.make_sudoku_constraint(p, 3)

In [5]: s = sudoku.dict_to_sudoku_string(c.getSolution(), 3)

In [6]: sudoku.print_puzzle(s, 3)
 2 5 8 7 3 6 9 4 1 
 6 1 9 8 2 4 3 5 7 
 4 3 7 9 1 5 2 6 8 
 3 9 5 2 7 1 4 8 6 
 7 6 2 4 9 8 1 3 5 
 8 4 1 6 5 3 7 2 9 
 1 8 4 3 6 9 5 7 2 
 5 7 6 1 4 2 8 9 3 
 9 2 3 5 8 7 6 1 4 

In [7]: lp = sudoku.lp_puzzle(sudoku.sudoku_string_to_dict(p), 3)

In [8]: s = sudoku.solve_as_lp_(lp, 3)

In [9]: ss = sudoku.dict_to_sudoku_string(s,3)

In [10]: sudoku.print_puzzle(ss, 3)
 2 5 8 7 3 6 9 4 1 
 6 1 9 8 2 4 3 5 7 
 4 3 7 9 1 5 2 6 8 
 3 9 5 2 7 1 4 8 6 
 7 6 2 4 9 8 1 3 5 
 8 4 1 6 5 3 7 2 9 
 1 8 4 3 6 9 5 7 2 
 5 7 6 1 4 2 8 9 3 
 9 2 3 5 8 7 6 1 4 

In [11]: setup_string = "from sudoku import empty_puzzle"

In [12]: experiment_string = """\
   ....: p = empty_puzzle(2)
   ....: s = p.getSolutions()
   ....: print len(s)"""

In [13]: from timeit import Timer

In [14]: t = Timer(experiment_string, setup_string)

In [15]: print t.timeit(1)
288
0.146998882294

