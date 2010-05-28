import sudoku

p="""           
25..3.9.1
.1...4...
4.7...2.8
..52.....
....981..
.4...3...
...36..72
.7......3
9.3...6.4
"""

c = sudoku.make_sudoku_constraint(p, 3)
s = sudoku.dict_to_sudoku_string(c.getSolution(), 3)
sudoku.print_puzzle(s, 3)

