:author: Sean Davis
:email: Sean_Davis@berea.edu
:institution: Berea College

:author: Matthew Henderson
:email: Matthew_Henderson@berea.edu
:institution: Berea College

:author: Andrew Smith
:email: Andrew_Smith@berea.edu
:institution: Berea College

------------------------------------------------
Modeling Sudoku puzzles with Python
------------------------------------------------

.. class:: abstract

   A short version of the long version that is way too long to be written as a
   short version anyway.

Introduction
------------

Everyone is familiar with Sudoku puzzles, which appear in newspapers daily the world over. A typical puzzle is shown in Figure XXX. To complete the puzzle requires the puzzler to fill the empty cells with numbers XXX in such a way as to have exactly one of every number in every row, every column and every of the small 3 by 3 boxes.

A well-formed Sudoku puzzle has a unique solution. This means that the puzzle can be solved by logic alone, without any guessing.

Sudoku puzzles have a variety of different difficulty levels. Harder puzzles typically have fewer prescribed symbols. It is unknown to this day how few cells need to be filled for a Sudoku puzzle to have a unique solution. Well-formed Sudoku with 17 symbols exist. It is unknown whether or not there exists a well-formed puzzle with 16 clues.

The authors have written an open-source library for modeling Sudoku puzzles in a variety of different mathematical domains. The source-code for XXX is available at XXX.

Cells in the Sudoku puzzle are represented by integers. The cell in row XXX and column XXX of a puzzle of dimension XXX with XXX rows is represented by the integer XXX. Standard puzzles have XXX rows so the integer is XXX.

In practice, the user mainly interacts with XXX either by creating specific puzzles instances through input of puzzle strings, directly or from a text file, or by using generator functions. 

For example, the puzzle dictionary in Figure XXX can be built from a puzzle string through use of the XXX function.

Or a random puzzle can be built by using the XXX function.

XXX random puzzle demo listing XXX

Simple functions are provided to access certain parameters associated with a puzzle.

The main power behind XXX, however, is the modeling capability of the library. In the next section we introduce the different modeling concepts and show how to use existing Python components to build models of Sudoku puzzles.

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


