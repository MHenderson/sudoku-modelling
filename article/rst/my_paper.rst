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

   The popular Sudoku puzzles which appear daily in newspapers the world over have, lately, attracted the attention of mathematicians and computer scientists. There are many, difficult, unsolved problems about Sudoku puzzles and their generalizations which make them especially interesting to mathematicians. Also, as is well-known, the generalization of the Sudoku puzzle to larger dimension is an NP-complete problem and therefore of substantial interest to computer scientists.

   In this article we discuss the modeling of Sudoku puzzles in a variety of different mathematical domains. We show how to use, already available, Python libraries and components to implement these models and how to use available solvers to find solutions to the models and translate those solutions back into Sudoku puzzle solutions. The models we discuss, which include translations to problems in the domains of constraint satisfaction, integer/linear programming, polynomial calculus and graph theory, are available in an open-source Python library ``sudoku.py`` developed by the authors.

Introduction
------------

Sudoku puzzles
~~~~~~~~~~~~~~

Everyone is familiar with Sudoku puzzles, which appear daily in newspapers the world over. A typical puzzle is shown in Figure XXX. 

XXX Figure 1 XXX

To complete the puzzle in Figure XXX requires the puzzler to fill every empty cell with an integer between 1 and 9 in such a way that every number from 1 up to 9 appears once in every row, every column and every one of the small 3 by 3 boxes highlighted with thick borders.

Sudoku puzzles have a variety of different difficulty levels. Harder puzzles typically have fewer prescribed symbols. A well-formed Sudoku puzzle has a unique solution. This means that the puzzle can be solved by logic alone, without any guessing. It is unknown to this day how few cells need to be filled for a Sudoku puzzle to be well-formed. Well-formed Sudoku with 17 symbols exist. It is unknown whether or not there exists a well-formed puzzle with only 16 clues.

A few words about terminology and notation in this paper. By *Sudoku puzzle of boxsize* :math:`$n$` is meant a partial assignment of :math:`$n^2$` values to the cells of an :math:`n^2 \times n^2$` grid in such a way that at most one of each symbols occurs in any row, column or box. A *solution* of a Sudoku puzzle is a complete assignment to the cells, satisfying the same conditions on row, columns and boxes, which agrees with the partial assignment.

sudoku.py
~~~~~~~~~

The authors have written an open-source Python library ``sudoku.py`` for modeling Sudoku puzzles in a variety of different mathematical domains. The source-code for ``sudoku.py`` is available at `http://bitbucket.org/matthew/scipy2010 <http://bitbucket.org/matthew/scipy2010>`_.

With this library, the process of building models of Sudoku puzzles, which can then be solved using algorithms for computing solutions of the models, is a simple matter. In order to understand how to build the models, first it is necessary to understand how Sudoku puzzles are represented in ``sudoku.py``.

There are two different representations which the user should be aware. Puzzles can be represented either as strings, or dictionaries. 

The dictionary representation of a puzzle is a mapping between cell labels and cell values. Cell values are integers in the range :math:`$\{1, \ldots, n^2\}$` and cell labels are integers in the range :math:`$\{1, \ldots, n^4\}$`. The labeling of a Sudoku puzzle of boxsize :math:`$n$` starts with 1 in the top-left corner and moves along rows, continuing to the next row when a row is finished. So, the cell in row :math:`$i$` and column :math:`$j$` is labeled :math:`$(i - 1)n^2 + j$`.  

For example, the puzzle shown in Figure XXX is represented by the dictionary ::

    >>> d = {1: 2, 2: 5, 5: 3, 7: 9, 9: 1,
    ...     11: 1, 15: 4, 19: 4, 21: 7, 25: 2,
    ...     27: 8, 30: 5, 31: 2, 41: 9, 42: 8,
    ...     43: 1, 47: 4, 51: 3, 58: 3, 59: 6,
    ...     62: 7, 63: 2, 65: 7, 72: 3, 73: 9,
    ...     75: 3, 79: 6, 81: 4}

A Sudoku puzzle object can be built from such a dictionary. Note that the boxsize is a parameter of the ``Puzzle`` object. ::
 
    >>> import sudoku
    >>> p = sudoku.Puzzle(d, 3)
    >>> p
     2  5  .  .  3  .  9  .  1 
     .  1  .  .  .  4  .  .  . 
     4  .  7  .  .  .  2  .  8 
     .  .  5  2  .  .  .  .  . 
     .  .  .  .  9  8  1  .  . 
     .  4  .  .  .  3  .  .  . 
     .  .  .  3  6  .  .  7  2 
     .  7  .  .  .  .  .  .  3 
     9  .  3  .  .  .  6  .  4 

In practice, however, the user mainly interacts with ``sudoku.py`` either by creating specific puzzles instances through input of puzzle strings, directly or from a text file, or by using generator functions. 

The string representation of a Sudoku puzzle of boxsize :math:`$n$` is a string of ascii characters of length :math:`$n^4$`. In such a string a period character represents an empty cell and other ascii characters are used to specify assigned values. Whitespace characters and newlines are ignored when ``Puzzle`` objects are built from strings.

A possible string representation of the puzzle in Figure XXX is: ::
    
    >>> s = """
    ... 2 5 . . 3 . 9 . 1
    ... . 1 . . . 4 . . .
    ... 4 . 7 . . . 2 . 8
    ... . . 5 2 . . . . .
    ... . . . . 9 8 1 . .
    ... . 4 . . . 3 . . .
    ... . . . 3 6 . . 7 2
    ... . 7 . . . . . . 3
    ... 9 . 3 . . . 6 . 4
        """

By specifying that the input is a string (with a keyword parameter `format = 's'`) a puzzle object can be built from a puzzle string. ::

    >>> import sudoku
    >>> p = sudoku.Puzzle(s, 3, format = 's')

XXX Reading puzzle strings from files XXX

Random puzzles can be created by the ``random_puzzle`` function. ::

    >>> q = sudoku.random_puzzle(15, 3)
    >>> q
     .  .  .  .  5  .  .  .  1 
     .  5  .  .  .  .  .  .  7 
     .  .  1  9  .  7  .  .  . 
     .  .  .  .  .  .  .  .  . 
     .  .  5  .  .  .  7  .  . 
     .  .  6  .  .  .  .  9  . 
     .  .  .  .  .  5  .  .  . 
     5  .  .  .  .  .  4  .  . 
     1  .  .  .  .  .  .  .  . 

The first argument to ``random_puzzle`` is the number of prescribed cells in the puzzle.    

Solving of puzzles is handled by the ``solve`` function. This function can use a variety of different algorithms, specified by an optional keyword argument, to solve the puzzle. The default behavior is to use a constraint propagation algorithm. ::

    >>> s = sudoku.solve(q)
    >>> s
     7  3  2  8  5  6  9  4  1 
     8  5  9  4  2  1  6  3  7 
     6  4  1  9  3  7  8  5  2 
     9  7  8  5  4  3  1  2  6 
     3  2  5  6  1  9  7  8  4 
     4  1  6  7  8  2  5  9  3 
     2  9  4  1  6  5  3  7  8 
     5  6  3  2  7  8  4  1  9 
     1  8  7  3  9  4  2  6  5 

Puzzles of boxsize other than 3 can also be modeled with ``sudoku.py``. ::

    >>> q2 = sudoku.random_puzzle(7, 2)
    >>> q2
     4  .  .  . 
     2  1  .  . 
     .  4  .  2 
     .  .  3  4
    >>> sudoku.solve(q2)
     4  3  2  1 
     2  1  4  3 
     3  4  1  2 
     1  2  3  4 
    >>> q4 = sudoku.random_puzzle(200, 4)
    >>> q4
     .  .  e  d  .  .  a  9  8  .  .  5  .  3  2  1 
     c  b  a  9  4  .  2  1  g  .  e  d  8  7  6  . 
     8  .  6  5  g  f  e  d  4  3  2  1  c  b  a  9 
     .  .  2  1  8  7  6  5  c  .  a  .  g  f  e  d 
     f  d  g  .  9  8  7  c  3  6  .  b  .  2  .  . 
     2  6  .  .  1  d  g  b  f  4  c  .  9  .  8  7 
     .  4  1  8  3  6  .  2  9  e  7  .  .  .  5  c 
     9  c  7  b  e  a  5  .  2  1  .  8  f  g  3  6 
     e  g  9  f  7  .  8  a  6  d  3  4  5  1  b  . 
     b  a  .  7  .  2  9  e  5  .  1  f  .  8  c  . 
     3  8  .  6  5  1  4  f  .  9  b  2  7  a  d  g 
     .  .  4  .  d  g  b  3  7  a  8  c  e  6  9  f 
     .  e  f  c  2  9  3  8  a  5  g  7  6  4  .  b 
     7  9  .  4  a  .  1  6  d  8  .  e  2  c  g  3 
     6  2  8  g  b  .  d  .  .  c  9  3  .  .  f  . 
     5  1  3  a  f  e  c  g  b  2  4  6  .  .  7  8 
     >>> sudoku.solve(q4)
     g  f  e  d  c  b  a  9  8  7  6  5  4  3  2  1 
     c  b  a  9  4  3  2  1  g  f  e  d  8  7  6  5 
     8  7  6  5  g  f  e  d  4  3  2  1  c  b  a  9 
     4  3  2  1  8  7  6  5  c  b  a  9  g  f  e  d 
     f  d  g  e  9  8  7  c  3  6  5  b  1  2  4  a 
     2  6  5  3  1  d  g  b  f  4  c  a  9  e  8  7 
     a  4  1  8  3  6  f  2  9  e  7  g  b  d  5  c 
     9  c  7  b  e  a  5  4  2  1  d  8  f  g  3  6 
     e  g  9  f  7  c  8  a  6  d  3  4  5  1  b  2 
     b  a  d  7  6  2  9  e  5  g  1  f  3  8  c  4 
     3  8  c  6  5  1  4  f  e  9  b  2  7  a  d  g 
     1  5  4  2  d  g  b  3  7  a  8  c  e  6  9  f 
     d  e  f  c  2  9  3  8  a  5  g  7  6  4  1  b 
     7  9  b  4  a  5  1  6  d  8  f  e  2  c  g  3 
     6  2  8  g  b  4  d  7  1  c  9  3  a  5  f  e 
     5  1  3  a  f  e  c  g  b  2  4  6  d  9  7  8 
   
Models
------

The main power behind ``sudoku.py`` is the modeling capability of the library. In this section we introduce several models of Sudoku and show how to use existing Python components to build models of Sudoku puzzles. The models introduced here are all implemented in ``sudoku.py``. Implementations are discussed and demonstrations of the components of ``sudoku.py`` corresponding to each of the different models are given. 

Constraint models
~~~~~~~~~~~~~~~~~

Constraint models for Sudoku puzzles are discussed in [Sim05]_. A simple model uses the AllDifferent constraint.

A constraint program is a collection of constraints. A constraint restricts the values which can be assigned to certain variables in a solution of the constraint problem. The AllDifferent constraint restricts variables to having mutually different values. 

Modeling Sudoku puzzles is easy with the AllDifferent constraint. To model the empty Sudoku puzzle (i.e. the puzzle with no clues) we simply form a constraint program having an AllDifferent constraint for every row, column and box.

For example, if we let :math:`$x_{i} \in \{1,\ldots,n^2\}$` for :math:`$1 \leq i \leq n^4$`, where :math:`$x_{i} = j$` means that cell :math:`$i$` gets value :math:`$j$` then the constraint model for a Sudoku puzzle of boxsize :math:`$n = 3$` would include constraints

.. raw:: latex

   \[\mathrm{AllDifferent}(x_{1}, x_{2}, x_{3}, x_{4}, x_{5}, x_{6}, x_{7}, x_{8}, x_{9})\]
   \[\mathrm{AllDifferent}(x_{1}, x_{10}, x_{19}, x_{28}, x_{37}, x_{46}, x_{55}, x_{64}, x_{73})\]
   \[\mathrm{AllDifferent}(x_{1}, x_{2}, x_{3}, x_{10}, x_{11}, x_{12}, x_{19}, x_{20}, x_{21})\]

to constrain, respectively, the variables in the first row, column and box.

The Sudoku constraint model in ``sudoku.py`` is implemented using ``python-constraint v1.1`` by Gustavo Niemeyer. This open-source library is available at `http://labix.org/python-constraint <http://labix.org/python-constraint>`_.

With ``python-constraint`` we create a ``Problem`` instance which has variables for each element of ``cells(n)``. Each variable has the same domain ``symbols(n)`` of possible symbols. The ``Problem`` member function ``addVariables`` provides a convenient method for adding variables to a constraint problem object. ::

    >>> from constraint import Problem
    >>> from sudoku import cells, symbols
    >>> p = Problem()
    >>> p.addVariables(cells(n), symbols(n))
 
The AllDifferent constraint in ``python-constraint`` is implemented  as ``AllDifferentConstraint()``. The ``addConstraint(constraint, variables)`` member function is used to add a ``constraint`` on ``variables`` to a constraint problem object. So, to build an empty Sudoku puzzle constraint model we can do the following. ::
    
    >>> from constraint import AllDifferentConstraint
    >>> from sudoku import \
    ...   cells_by_row, cells_by_col, cells_by_box
    >>> for row in cells_by_row(n):
    ...   p.addConstraint(AllDifferentConstraint(), row)
    >>> for col in cells_by_col(n):    
    ...   p.addConstraint(AllDifferentConstraint(), col)
    >>> for box in cells_by_box(n):
    ...   p.addConstraint(AllDifferentConstraint(), box)

Here the functions ``cells_by_row``, ``cells_by_col`` and ``cells_by_box`` give the cell labels of a Sudoku puzzle ordered, respectively, by row, column and box. These three loops, respectively, add to the constraint problem object the necessary constraints on row, column and box variables.

To extend this model to a Sudoku puzzle with clues requires additional constraints to ensure that the values assigned to clue variables are fixed. One possibility is to add an ExactSum constraint for each clue. 

The ExactSum constraint restricts the sum of a set of variables to a precise given value. We can exploit the ExactSum constraint to specify that certain individual variables are given certain specific values. In particular, if the puzzle clues are given by a dictionary ``fixed`` then we can complete our model by adding the following constraints. ::

    >>> from constraint import ExactSumConstraint as Exact
    >>> for cell in fixed:
    ...   p.addConstraint(Exact(fixed[cell]), [cell])

To solve the Sudoku puzzle given by the ``fixed`` dictionary now can be done by solving the constraint model ``p``. The constraint propogation algorithm of ``python-constraint`` can be invoked by the ``getSolution`` member function. ::

    >>> s = p.getSolution()
    >>> s
     2  5  8  7  3  6  9  4  1 
     6  1  9  8  2  4  3  5  7 
     4  3  7  9  1  5  2  6  8 
     3  9  5  2  7  1  4  8  6 
     7  6  2  4  9  8  1  3  5 
     8  4  1  6  5  3  7  2  9 
     1  8  4  3  6  9  5  7  2 
     5  7  6  1  4  2  8  9  3 
     9  2  3  5  8  7  6  1  4

The general ``solve`` function provided by ``sudoku.py`` knows how to build a constraint model like above, solve it and translate the solution into a completed Sudoku puzzle. ::

    >>> s = sudoku.solve(d, 3, model = 'CP')

In fact, the model keyword argument in this case is redundant, as 'CP' is the default value.

Graph models
~~~~~~~~~~~~

A graph model for Sudoku is presented in [Var05]_. In this model, every cell of the Sudoku grid is represented by a node of the graph. The edges of the graph are given by the dependency relationships between cells. In other words, if two cells lie in the same row, column or box, then their nodes are joined by an edge in the graph.

XXX Example: The Shidoku graph XXX

In the graph model, a Sudoku puzzle is given by a partial assignment of colors to the nodes of the graph. The color assigned to a node corresponds to a value assigned to the corresponding cell. A solution of the puzzle is given by a coloring of the nodes with :math:`$n^2$` colors which extends the original partial coloring. A node coloring of the Sudoku graph which corresponds to a completed puzzle has the property that adjacent vertices are colored differently. Such a node coloring is called 'proper'.

XXX Example: Solving a Shidoku puzzle via node coloring the graph model. XXX

The Sudoku graph model in ``sudoku.py`` is implemented using ``networkx v1.1``. This open-source Python graph library is available at `http://networkx.lanl.gov/ <http://networkx.lanl.gov/>`_ ::

    >>> import networkx
    >>> g = networkx.Graph()

Modeling an empty Sudoku puzzle as a ``networkx`` Graph object requires adding nodes for every cell and edges for every pair of dependent cells. To add nodes (respectively, edges) to a graph, ``networkx`` provides graph member functions ``add_nodes_from`` (respectively, ``add_edges_from``). Cell labels can be obtained from ``sudoku.py``'s ``cells`` function. ::

    >>> g.add_nodes_from(sudoku.cells(n))

Dependent cells can be computed using the ``dependent_cells`` function, which returns a list of all pairs :math:`$(x, y)$` with :math:`$x < y$` such that :math:`$x$` and :math:`$y$` either lie in the same row, same column or same box.  ::

    >>> g.add_edges_from(sudoku.dependent_cells(n))

To model a Sudoku puzzle, we have to be able to assign colors to vertices. Graphs in ``networkx`` allow arbitrary data to be associated with graph nodes. Again, suppose that ``fixed`` is a dictionary of puzzle clues. ::

    >>> for cell in fixed:
    ...   g.node[cell]['color'] = fixed[cell]

There are many vertex coloring algorithms which we can use to try to find a solution of a puzzle. In ``sudoku.py``, a general vertex coloring algorithm is implemented. This generalized algorithm can be customized to provide a variety of different specific algorithms. 

To use the graph model to find a solution to the Sudoku puzzle of Figure XXX, we can again call the ``solve`` function, specifying ``graph`` as the model. ::

    >>> s = sudoku.solve(d, 3, model = 'graph')

Polynomial system models
~~~~~~~~~~~~~~~~~~~~~~~~

The graph model above is introduced in [Var05]_ as a prelude to modeling Sudoku puzzles as systems of polynomial equations. The polynomial system model in [Var05]_ involves variables :math:`$x_{i}$` for :math:`i \in \{1,\ldots,n^4\}` where :math:`$x_{i} = j$` is interpreted as the cell with label :math:`$i$` being assigned the value :math:`$j$`.

The Sudoku polynomial-system model in sudoku.py is implemented using ``sympy v0.6.7``. This open-source symbolic algebra Python library is available at `http://code.google.com/p/sympy/ <http://code.google.com/p/sympy/>`_

Variables in ``sympy`` are ``Symbol`` objects. A ``sympy.Symbol`` object has a name. So, to construct the variables for our model, first we map symbol names on to each cell label. ::

    def cell_symbol_names(n):
      return map(lambda i:'x' + str(i), cells(n))

Now, with these names for the symbols which represent the cells of our Sudoku puzzle, we can construct the cell variable symbols themselves. ::

    def cell_symbols(n):
      return map(sympy.Symbol, cell_symbol_names(n))

Now, using these variables, we can build a Sudoku polynomial system model. The polynomial system model presented in [Var05]_ is based on the graph model of the previous section. There are polynomials in the system for every node in the graph model and polynomials for every edge. 

The role of the node polynomials in the polynomial system is to ensure that every variable is assigned a number from :math:`$\{1,\ldots,n^2\}$` in every solution:

.. raw:: latex

   \[F(x_{j}) = \prod_{i = 1}^{n^{2}} (x_{j} - i)\]

Node polynomials, for ``sympy.Symbol`` ``x`` can be built as follows. (Here ``symbols`` is just a function which returns the list :math:`${1,\ldots,\n^2\}$`.) ::

    >>> from operator import mul
    >>> from sudoku import symbols
    >>> def F(x,n):
          return reduce(mul,[(x-s) for s in symbols(n)])

Edge polynomials, for adjacent vertices :math:`$x_i$` and :math:`$x_j$`, have the form: 

.. raw:: latex

   \[G(x_{i}, x_{j}) = \frac{F(x_{i}) - F(x_{j})}{x_{i} - x_{j}}\]

In ``sympy``, we build edge polynomials from the node polynomial function. ::

   >>> from sympy import cancel, expand
   >>> def G(x,y,n):
         return expand(cancel((F(x,n)-F(y,n))/(x-y)))

The polynomial model for the empty Sudoku puzzle now consists of the collection of all node polynomials for nodes in the Sudoku graph and all edge polynomials for node pairs ``(x,y)`` in ``dependent_symbols(n)``, where the ``dependent_symbols`` function maps ``sympy.Symbol`` onto the list of dependent cells.

To specify a Sudoku puzzle requires adding polynomials to represent the clues, or fixed cells. According to the model from [Var05]_, if :math:`$F$` is the set of fixed cells (i.e. cell label, value pairs) then to the polynomial system we need to add polynomials 
   
.. raw:: latex

   \[C(x_i, j) = x_i - j\]

Or, with ``sympy``: ::

    >>> def C(i, j):
    ...   sympy.Symbol('x' + str(i)) - j

In Listing XXX, an example is shown of how to use the polynomial-system model to find a solution to the Sudoku puzzle of Figure XXX. ::

    >>> s = sudoku.solve(d, 3, model = 'groebner')

Integer programming models
~~~~~~~~~~~~~~~~~~~~~~~~~~

In [Bar08]_ a model of Sudoku as an integer programming problem is presented. In this model, the variables are all binary.

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
    \sum_{j = mq - m + q}^{mq} \sum_{i = mp - m + 1}^{mp} x_{ijk} = 1
   \]
   \[
    1 \leq k \leq n, 1 \leq p \leq m, 1 \leq q \leq m
   \]   

The Sudoku integer programming model is implemented in ``sudoku.py`` using ``pyglpk v0.3`` by Thomas Finley. This open-source mixed integer/linear programming Python library is available at `http://tfinley.net/software/pyglpk/ <http://tfinley.net/software/pyglpk/>`_ 

In ``pyglpk``, an integer program is represented by a matrix of coefficients of the linear equations which make up the integer program. Two functions of ``sudoku.py`` provide the correct dimensions of the coefficient matrix. ::

    >>> import glpk
    >>> lp = glpk.LPX()
    >>> lp.cols.add(lp_matrix_ncols(n))
    >>> lp.rows.add(lp_matrix_nrows(n))

Columns of the matrix represent different variables. All our variables are binary and so their boundaries are set appropriately: ::

    >>> for c in lp.cols:
    ...   c.bounds = 0.0, 1.0

Rows of the coefficient matrix represent different linear equations. We require all our equations to have a value of 1, so we set the lower and upper bound of every equation to be 1. ::

    >>> for r in lp.rows:
    ...   r.bounds = 1.0, 1.0

With appropriate dimensions and bounds fixed, the coefficient matrix itself is provided by ``sudoku.py``'s ``lp_matrix`` function. ::

    >>> lp.matrix = lp_matrix(n)

To solve the original integer programming problem requires first solving a linear relaxation of the model. A solution of the linear relaxation is obtained by using the simplex algorithm provided by ``pyglpk`` :: 

    >>> lp.simplex()

Once the linear relaxation is solved, the original integer program can be solved. ::

    >>> for col in lp.cols:
    ...   col.kind = int
    >>> lp.integer()

Finally, we need to extract the solution as a dictionary from the model: ::

    >>> d = lp_to_dict(lp, n)
    >>> s = sudoku.Puzzle(d, 3)
    >>> s
     2  5  8  7  3  6  9  4  1 
     6  1  9  8  2  4  3  5  7 
     4  3  7  9  1  5  2  6  8 
     3  9  5  2  7  1  4  8  6 
     7  6  2  4  9  8  1  3  5 
     8  4  1  6  5  3  7  2  9 
     1  8  4  3  6  9  5  7  2 
     5  7  6  1  4  2  8  9  3 
     9  2  3  5  8  7  6  1  4

In Listing XXX, an example is shown of how to use the integer programming model to find a solution to the Sudoku puzzle of Figure XXX. ::

    >>> s = sudoku.solve(d, 3, model = 'lp')

Experimentation
---------------

In this section we demonstrate how to use ``sudoku.py`` to create Python scripts for experimentation with Sudoku puzzles. For the purposes of demonstration, we reproduce several results from the literature. The enumeration of Shidoku puzzles, coloring of the Sudoku graph, investigatations into minimally uniquely completable Sudoku puzzles and random puzzles.

Really, the aim of this section is to show how ``sudoku.py`` makes the task of writing these experimental investigation scripts very easy.

Enumerating Shidoku
~~~~~~~~~~~~~~~~~~~

Enumeration of Sudoku puzzles is a very difficult computational problem, which has been solved by XXX in YYY. The enumeration of Shidoku, however, is easy. To solve the enumeration problem for Shidoku, using the constraint model implemented in ``sudoku.py``, takes only a few lines of code and a fraction of a second of computation. ::

    >>> s = "from sudoku import Puzzle, count_solutions"
    >>> e = "print count_solutions(Puzzle({}, 2))"
    >>> from timeit import Timer
    >>> t = Timer(e, s)
    >>> print t.timeit(1)
    288
    0.146998882294

Coloring the Sudoku graph
~~~~~~~~~~~~~~~~~~~~~~~~~

As discussed above in the section on "Graph models", a completed Sudoku puzzle is equivalent to a minimal proper node coloring of the Sudoku graph. 

We have experimented with several different node coloring algorithms to see which are more effective, with respect to number of colors, at coloring the Sudoku graph. 

At first, we used Joseph Culberson's graph coloring programs (`http://webdocs.cs.ualberta.ca/~joe/Coloring/index.html <http://webdocs.cs.ualberta.ca/~joe/Coloring/index.html>`_) by writing the graph information to a file in Dimacs format (via the ``dimacs_string`` function of ``sudoku.py``. 

Of the programs we experimented with, the program implementing the saturation degree (DSatur) algorithm of XXX from YYY proved most effective in acheiving the minimal number of colors. 

In ``sudoku.py`` we have implemented a general graph coloring algorithm directly in Python which can reproduce the DSatur algorithm as well as several other node coloring algorithms.

The vertex coloring algorithm of ``sudoku.py``, ``vertex_coloring``, has keyword parameters ``nodes`` and ``choose_color`` which allow for customization of the general scheme. The ``nodes`` parameter specifies an ordering of vertices while ``choose_color`` is a visitor object for selecting the color of an uncolored vertex.

For example, if ``nodes`` is assigned the ``InOrder`` class and ``choose_color`` the ``first_available_color`` function then ``vertex_coloring(graph, nodes, choose_color)`` is the basic sequential vertex coloring algorithm.

We can get a random order by assigning ``RandomOrder`` to ``nodes``.

The DSatur algorithm is obtained by choosing ``DSATOrder`` for ``nodes`` and ``first_available_color`` for ``choose_color``.

XXX DSatur algorithm is an online algorithm XXX supported because ``nodes`` is an iterator.

The following table compares the DSatur algorithm with both sequential and random node coloring algorithms on a selection of Sudoku puzzles (obtained from XXX).

XXX Table of average, minimum and maximum number of colors used on some puzzles XXX

Minimal uniquely completable puzzles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gordon Royle maintains a list of uniquely completable 17-hint Sudoku puzzles at `http://mapleta.maths.uwa.edu.au/~gordon/sudoku17 <http://mapleta.maths.uwa.edu.au/~gordon/sudoku17>`_

XXX With ``sudoku.py`` we can write a short script for processing such puzzle collections XXX

XXX emphasize verifiction of uniqueness XXX

Hardness of random puzzles
~~~~~~~~~~~~~~~~~~~~~~~~~~

We have introduced the ``random_puzzle`` function in the introduction. The method by which this function produces a random puzzle is fairly simple. A completed Sudoku puzzle is first generated by solving the empty puzzle and then from this completed puzzle the appropriate number of clues is removed.

An interesting problem is to investigate the behavior of our different models on random puzzles. A simple script, available in the `investigations` folder of the source code, has been written to time the solution of models of random puzzles and plot the timings.

Two plots produced by this script highlight the different behavior of the constraint model and the integer programming model.

The first plot has time on the vertical axis and the number of clues on the horizontal axis. From this plot it seems that the constraint propogation algorithm finds puzzles with many or few clues easy. The difficult problems for the constraint solver appear to be clustered in the range of 20 to 30 clues.

 .. image:: random_CP.png

A different picture emerges with the linear programming model. With the same set of randomly generated puzzles it appears that, in general, the more clues the easier the solver finds a solution.

 .. image:: random_lp.png

References
----------
.. [Bar08] A. Bartlett, T. Chartier, A. Langville, T. Rankin. *An Integer Programming Model for the Sudoku Problem*,
           J. Online Math. & Its Appl., 8(May 2008), May 2008
.. [Var05] J. Gago-Vargas, I. Hartillo-Hermosa, J. Martin-Morales, J. M. Ucha- Enriquez, *Sudokus and Groebner Bases: not only a Divertimento*,
           XXXXXXXXXXXXXXXX 2005
.. [Lew05] R. Lewis. *Metaheuristics can solve Sudoku puzzles*,
           XXXXXXXXXXXXXXXX 2005
.. [Sim05] H. Simonis. *Sudoku as a Constraint Problem*, 
           XXXXXXXXXXXXXXXX 2005
.. [Nie05] G. Niemeyer. *python-constraint*,
           XXXXXXXXXXXXXXXX
.. [Fin09] T. Finley. *pyglpk*,
           XXXXXXXXXXXXXXXXXXX
.. [Ntx10] Networkx Developers, *networkx*,
           XXXXXXXXXXXXXXXXXXX
.. [Sym10] sympy developers, *sympy*,
           XXXXXXXXXXXXXXXXXXX

