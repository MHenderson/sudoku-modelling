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

Models
------

In this section we introduce several models of Sudoku. The models introduced here are implemented in the open-source library (\texttt{sudoku.py}) developed by the authors. Demonstrations of the library components corresponding to each of the different models are given.

Constraint models
~~~~~~~~~~~~~~~~~

Constraint models for Sudoku puzzles are discussed in XXX. The simplest model uses the all_different constraint.

In Listing XXX, an example is shown of how to model a Sudoku puzzle with XXX and use the built-in constraint solver of XXX to find a solution.

Graph models
~~~~~~~~~~~~

A graph model for Sudoku is presented in XXX. In this model, every cell of the Sudoku grid is represented by a vertex. The edges of the graph are given by the cell dependency relations. In other words, if two cells lie in the same row, column or box, then their vertices are joined by an edge in the graph model.

Polynomial system models
~~~~~~~~~~~~~~~~~~~~~~~~

The graph model in XXX can be used to model a Sudoku puzzle as a system of polynomial equations. The polynomial system model presented in XXX consists of a polynomial for every vertex in the graph model and a polynomial for every edge. The vertex polynomials have the form :math:`$F(x_j) = \prod_{i=1}^{9} (x_j - i)$`. The edge polynomials are :math:`$G(x_i, x_j) = \frac{F(x_i) - F(x_j)}{x_i - x_j}$`, where :math:`$x_i$` and :math:`$x_j$` are adjacent vertices in the graph model. 

Integer programming models
~~~~~~~~~~~~~~~~~~~~~~~~~~

In XXX a model of Sudoku as an integer programming problem is presented. In this model, the variables are all binary.

.. raw:: latex 

   \[x_{ijk} \in \{0, 1\}\]


Variable :math:`$x_{ijk}$` represents the assignment of symbol :math:`$k$` to cell :math:`$(i,j)$` in the Sudoku puzzle.

.. raw:: latex

   \[
    x_{ijk} = 
     \left\lbrace 
      \begin{array}{rl}
       1 & \mbox{ if cell $(i, j)$ contains symbol $k$} \\
       0 & \mbox{ otherwise}
      \end{array}
     \right.
   \]

This model has a set of equations which force every solution to assign a symbol to every cell in the finished Sudoku puzzle.

.. raw:: latex

   \[
    \sum_{k = 1}^{n} x_{ijk} = 1, \quad 1 \leq i \leq n, 1 \leq j \leq n
   \]

Fixed elements in the Sudoku puzzle, given by a set :math:`$F$` of triples :math:`$(i,j,k)$`, are each represented by an equation in the system:   

.. raw:: latex

   \[
     x_{ijk} = 1, \quad \forall (i,j,k) \in F
   \]

The remaining equations in this model represent the unique occurence of every symbol in every column:

.. raw:: latex
   
   \[
    \sum_{i = 1}^{n} x_{ijk} = 1, \quad 1 \leq j \leq n, 1 \leq k \leq n
   \]

every symbol in every row:

.. raw:: latex
   
   \[
    \sum_{j = 1}^{n} x_{ijk} = 1, \quad 1 \leq i \leq n, 1 \leq k \leq n
   \]

and every symbol in every box:

.. raw:: latex

   \[
    \sum_{j = mq - m + q}^{mq} \sum_{i = mp - m + 1}^{mp} x_{ijk} = 1, \quad 1 \leq k \leq n, 1 \leq p \leq m, 1 \leq q \leq m
   \]   

References
----------
.. [Atr03] P. Atreides. *How to catch a sandworm*,
           Transactions on Terraforming, 21(3):261-300, August 2003.


