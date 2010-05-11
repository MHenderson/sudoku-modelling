# Sean Davis, Matthew Henderson, Andrew Smith (Berea) 4.1.2010

import sudoku

c = sudoku.make_sudoku_constraint("1234432131422410",2)

assert sudoku.cells(0)==[]
assert sudoku.cells(1)==[1]
assert sudoku.cells(2)==[1 ,2 ,3 ,4,
                         5 ,6 ,7 ,8,
                         9 ,10,11,12,
                         13,14,15,16]
assert sudoku.cells(3)==[1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 , 9,
                         10,11,12,13,14,15,16,17,18,
                         19,20,21,22,23,24,25,26,27,
                         28,29,30,31,32,33,34,35,36,
                         37,38,39,40,41,42,43,44,45,
                         46,47,48,49,50,51,52,53,54,
                         55,56,57,58,59,60,61,62,63,
                         64,65,66,67,68,69,70,71,72,
                         73,74,75,76,77,78,79,80,81]

assert sudoku.symbols(0)==[]
assert sudoku.symbols(1)==[1]
assert sudoku.symbols(2)==[1,2,3,4]
assert sudoku.symbols(3)==[1,2,3,4,5,6,7,8,9]

assert sudoku.top_left_cells(1)==[]
assert sudoku.top_left_cells(2)==[1,3,9,11]
assert sudoku.top_left_cells(3)==[1,4,7,28,31,34,55,58,61]

assert sudoku.rows(0)==[]
assert sudoku.rows(1)==[[1]]
assert sudoku.rows(2)==[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
assert sudoku.rows(3)==[[1, 2, 3, 4, 5, 6, 7, 8, 9], 
                        [10, 11, 12, 13, 14, 15, 16, 17, 18], 
                        [19, 20, 21, 22, 23, 24, 25, 26, 27], 
                        [28, 29, 30, 31, 32, 33, 34, 35, 36], 
                        [37, 38, 39, 40, 41, 42, 43, 44, 45], 
                        [46, 47, 48, 49, 50, 51, 52, 53, 54], 
                        [55, 56, 57, 58, 59, 60, 61, 62, 63], 
                        [64, 65, 66, 67, 68, 69, 70, 71, 72], 
                        [73, 74, 75, 76, 77, 78, 79, 80, 81]]

assert sudoku.cols(0)==[]
assert sudoku.cols(1)==[[1]]
assert sudoku.cols(2)==[[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]]
assert sudoku.cols(3)==[[1, 10, 19, 28, 37, 46, 55, 64, 73], 
                        [2, 11, 20, 29, 38, 47, 56, 65, 74], 
                        [3, 12, 21, 30, 39, 48, 57, 66, 75], 
                        [4, 13, 22, 31, 40, 49, 58, 67, 76], 
                        [5, 14, 23, 32, 41, 50, 59, 68, 77], 
                        [6, 15, 24, 33, 42, 51, 60, 69, 78], 
                        [7, 16, 25, 34, 43, 52, 61, 70, 79], 
                        [8, 17, 26, 35, 44, 53, 62, 71, 80], 
                        [9, 18, 27, 36, 45, 54, 63, 72, 81]]

assert sudoku.boxes(1)==[]
assert sudoku.boxes(2)==[[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,15,16]]
assert sudoku.boxes(3)==[[1, 2, 3, 10, 11, 12, 19, 20, 21], 
                         [4, 5, 6, 13, 14, 15, 22, 23, 24], 
                         [7, 8, 9, 16, 17, 18, 25, 26, 27], 
                         [28, 29, 30, 37, 38, 39, 46, 47, 48], 
                         [31, 32, 33, 40, 41, 42, 49, 50, 51], 
                         [34, 35, 36, 43, 44, 45, 52, 53, 54], 
                         [55, 56, 57, 64, 65, 66, 73, 74, 75], 
                         [58, 59, 60, 67, 68, 69, 76, 77, 78], 
                         [61, 62, 63, 70, 71, 72, 79, 80, 81]]

assert sudoku.cell_symbol_names(0)==[]
assert sudoku.cell_symbol_names(1)==['x1']
assert sudoku.cell_symbol_names(2)==['x1', 'x2', 'x3', 'x4',
                                     'x5', 'x6', 'x7', 'x8',
                                     'x9', 'x10','x11','x12',
                                     'x13','x14','x15','x16']
assert sudoku.cell_symbol_names(3)==[
                         'x1','x2','x3','x4','x5','x6','x7','x8','x9',
                         'x10','x11','x12','x13','x14','x15','x16','x17','x18',
                         'x19','x20','x21','x22','x23','x24','x25','x26','x27',
                         'x28','x29','x30','x31','x32','x33','x34','x35','x36',
                         'x37','x38','x39','x40','x41','x42','x43','x44','x45',
                         'x46','x47','x48','x49','x50','x51','x52','x53','x54',
                         'x55','x56','x57','x58','x59','x60','x61','x62','x63',
                         'x64','x65','x66','x67','x68','x69','x70','x71','x72',
                         'x73','x74','x75','x76','x77','x78','x79','x80','x81']

p = sudoku.empty_puzzle(2)
s = p.getSolutions()
assert len(s)==288
