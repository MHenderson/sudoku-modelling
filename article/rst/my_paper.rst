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

   The popular Sudoku puzzles which appear daily in newspapers the world over have, lately, attracted the attention of mathematicians and computer scientists. There are several, difficult, unsolved problems about these puzzles which make them especially interesting to mathematicians. And, as is well-known, the generalization of the Sudoku puzzle to larger dimension is an NP-complete problem and therefore is a problem of substantial interest to computer scientists.

   Already, significant research has been done investigating both the automated generation of Sudoku and other related puzzles as well as on automating the solution of such puzzles. In both of these areas, a particularly important line of investigation has been the translation of these puzzles into the domains of constraint satisfaction and satisfiability.

   In this article we discuss these, and other, translations and show how to use, already available, Python libraries and components to implement these translations and how to use off-the-shelf algorithmic components to solve the translated puzzles. The translations we discuss, which include translations to problems in the domains of constraint satisfaction, integer/linear programming, polynomial calculus and graph theory, are available in an open-source Python library developed by the authors.

   Our intention in this article is to demonstrate the benefits of Python as an experimental framework for computer scientists and mathematicians and how to integrate modeling components with components for collecting and analyzing data. We compare this approach to the more traditional use of command-line tools which is commonplace in the constraints and satisfiability communities for such work. In particular, we discuss the modeling of Sudoku as satisfiability problems and analyse the relative performance of these translations against the translations we propose.

Introduction
------------

Sudoku puzzles
~~~~~~~~~~~~~~

Everyone is familiar with Sudoku puzzles, which appear in newspapers daily the world over. A typical puzzle is shown in Figure XXX. 

XXX Figure 1 XXX

To complete the puzzle in Figure XXX requires the puzzler to fill every empty cells with an integer between 1 and 9. The puzzle is only complete if all the empty cells are filled in such a way that every number appears once in every row, every column and every of the small 3 by 3 boxes.

A well-formed Sudoku puzzle has a unique solution. This means that the puzzle can be solved by logic alone, without any guessing.

Sudoku puzzles have a variety of different difficulty levels. Harder puzzles typically have fewer prescribed symbols. It is unknown to this day how few cells need to be filled for a Sudoku puzzle to have a unique solution. Well-formed Sudoku with 17 symbols exist. It is unknown whether or not there exists a well-formed puzzle with 16 clues.

A few words about terminology. In this paper, a Sudoku 'puzzle' is understood to mean a partial assignment of :math:`$n^2$` values to the cells of an :math:`n^2 \times n^2$` grid in such a way that at most one of each symbols occurs in any row, column or box. A 'solution' of a Sudoku puzzle is a complete assignment to the cells, satisfying the same conditions on row, columns and boxes, which agrees with the partial assignment.

sudoku.py
~~~~~~~~~

The authors have written an open-source library ``sudoku.py`` for modeling Sudoku puzzles in a variety of different mathematical domains. The source-code for sudoku.py is available at `http://bitbucket.org/matthew/scipy2010 <http://bitbucket.org/matthew/scipy2010>`_.

With this library, the process of building models of Sudoku puzzles, which can then be solved using algorithms for computing solutions of the models, is a simple matter. In order to understand how to build the models, first it is necessary to understand how Sudoku puzzles are represented when using the library.

There are two different representations which the user should be aware. Puzzles can be represented either as strings, or dictionaries. 

The dictionary representation of a puzzle is a mapping between cell labels and cell values. Cell values are integers in the range :math:`$\{1, \ldots, n^2\}$`. Cell labels are also integers. The cell in row :math:`$i$` and column :math:`$j$` of a puzzle with :math:`$r$` rows is represented by the integer :math:`$(i - 1)r + j$`. Standard puzzles of boxsize :math:`$n$` have :math:`$n^2$` rows so the integer is :math:`$(i - 1)n^2 + j$`. 

So, for example, the puzzle shown in Figure XXX is represented by the dictionary ::

    >>> d = {1: 2, 2: 5, 5: 3, 7: 9, 9: 1,
    ...     11: 1, 15: 4, 19: 4, 21: 7, 25: 2,
    ...     27: 8, 30: 5, 31: 2, 41: 9, 42: 8,
    ...     43: 1, 47: 4, 51: 3, 58: 3, 59: 6,
    ...     62: 7, 63: 2, 65: 7, 72: 3, 73: 9,
    ...     75: 3, 79: 6, 81: 4}

The string representation of a Sudoku puzzle of boxsize :math:`$n$` is a string of ascii characters of length :math:`$n^4$`. The . character represents an empty cell and other ascii characters (to be precise characters from the list of printable characters in the Python string library) are used to specify assigned values.

So the string representation of the puzzle in Figure XXX is: ::
    
    >>> p = """
    ... 25..3.9.1
    ... .1...4...
    ... 4.7...2.8
    ... ..52.....
    ... ....981..
    ... .4...3...
    ... ...36..72
    ... .7......3
    ... 9.3...6.4
        """

In practice, the user mainly interacts with ``sudoku.py`` either by creating specific puzzles instances through input of puzzle strings, directly or from a text file, or by using generator functions. 

For example, the puzzle dictionary in Figure XXX can be built from a puzzle string above through use of the ``string_to_dict`` function. ::

    >>> import sudoku
    >>> d = sudoku.string_to_dict(p, 3)

A random puzzle, as a dictionary, can be built by using the ``random_puzzle`` function. ::

    >>> q = sudoku.random_puzzle(15, 3)

The first argument is the number of prescribed cells in the puzzle.    

Puzzles, or their solutions, can be displayed or output to a file using the ``print_puzzle`` function. ::

    >>> sudoku.print_puzzle(q, 3)
     .  .  .  .  .  .  3  .  . 
     .  .  .  3  .  .  .  8  7 
     .  .  .  9  .  .  .  5  . 
     .  .  .  .  .  .  .  .  . 
     .  .  .  .  .  .  .  .  6 
     .  .  .  5  .  .  .  .  . 
     7  .  .  8  .  .  .  .  . 
     .  .  .  .  .  3  4  7  . 
     .  .  .  7  .  9  .  .  .

The ``print_puzzle`` function has several optional arguments to control the output. The padding between cells, the end of row character and whether output should be to standard output or a file, can all be customized.

Solving of puzzles is handled by the ``solve`` function. This function can use a variety of different algorithms, specified by an optional keyword argument, to solve the puzzle. The default behavior is to use a constraint propagation algorithm. ::

    >>> s = sudoku.solve(q, 3)
    >>> sudoku.print_puzzle(s, 3)
     9  8  1  6  5  7  3  4  2 
     5  4  6  3  2  1  9  8  7 
     3  7  2  9  8  4  6  5  1 
     8  1  3  4  7  6  5  2  9 
     4  2  5  1  9  8  7  3  6 
     6  9  7  5  3  2  8  1  4 
     7  6  4  8  1  5  2  9  3 
     1  5  9  2  6  3  4  7  8 
     2  3  8  7  4  9  1  6  5

The library also provides functions for handling input of puzzles from text files. 
XXX file-handling example XXX

Models
------

The main power behind ``sudoku.py`` is the modeling capability of the library. In this section we introduce several models of Sudoku and show how to use existing Python components to build models of Sudoku puzzles. The models introduced here are all implemented in ``sudoku.py``. Implementations are discussed belwo and demonstrations of the components corresponding to each of the different models are given. 

Constraint models
~~~~~~~~~~~~~~~~~

Constraint models for Sudoku puzzles are discussed in [Sim05]_. The simplest model uses the ``all_different`` constraint.

A constraint model is a collection of constraints, which restrict certain variables to have certain values inside their domain. The ``all_different`` constraint requires that all variables specified as parameters to the constraint take different values. This makes modeling Sudoku puzzles easy. We have an ``all_different`` constraint on every row, column and box.

The Sudoku constraint model in ``sudoku.py`` is implemented using ``python-constraint v1.1`` by Gustavo Niemeyer. This open-source library is available at `http://labix.org/python-constraint <http://labix.org/python-constraint>`_ ::

    >>> from constraint import Problem
    >>> from sudoku import cells, symbols
    >>> p = Problem()
    >>> p.addVariables(cells(boxsize), symbols(boxsize))

``python-constraint`` implements the ``all_different`` constraint as ``AllDifferentConstraint()``. The ``addConstraint(constraint, variables)`` member function is used to add a  constraint on ``variables`` to a constraint problem object. So, to build an empty Sudoku puzzle constraint model we can do the following. ::
    
    >>> from constraint import AllDifferentConstraint
    >>> for row in sudoku.cells_by_row(boxsize):
    ...    p.addConstraint(AllDifferentConstraint(), row)
    >>> for col in sudoku.cells_by_col(boxsize):    
    ...    p.addConstraint(AllDifferentConstraint(), col)
    >>> for box in sudoku.cells_by_box(boxsize):
    ...    p.addConstraint(AllDifferentConstraint(), box)

To extend this model so that the clues are fixed we need to add an ExactSumConstraint for each clue. The ``exact_sum`` constraint restricts the value of a variable to a precise given value. ::

    >>> from constraint import ExactSumConstraint as Exact
    >>> for cell in fixed:
    ...    p.addConstraint(Exact(fixed[cell]), [cell])

To solve the Sudoku puzzle given by the ``fixed`` dictionary now can be done by solving the constraint model ``p``. The constraint propogation algorithm of ``python-constraint`` can be invoked by the ``getSolution`` member function. ::

    >>> s = p.getSolution()
    >>> sudoku.print_puzzle(s, 3)
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

A graph model for Sudoku is presented in [Var05]_. In this model, every cell of the Sudoku grid is represented by a vertex. The edges of the graph are given by the cell dependency relations. In other words, if two cells lie in the same row, column or box, then their vertices are joined by an edge in the graph model.

XXX The Shidoku graph XXX

A Sudoku puzzle is specified by a partial assignment of colors to the vertices of the graph and a solution to the puzzle is a minimal vertex coloring.

The Sudoku graph model in ``sudoku.py`` is implemented using ``networkx v1.1``. This open-source Python library is available at `http://networkx.lanl.gov/ <http://networkx.lanl.gov/>`_

Building a graph model of an empty Sudoku puzzle using ``networkx`` requires ading a node for each cell and an edge for each pair of dependent cells. The cell labels can be obtained from ``sudoku.py``'s ``cells`` function. The dependent cells can be computed using the ``dependent_cells`` function. The ``dependent_cells`` function returns a list of all pairs (x, y) with x < y such that x and y either lie in the same row, same column or same box. To add nodes and edges to a graph, ``networkx`` helpfully provides the graph member functions ``add_nodes_from`` and ``add_edges_from``. ::

    >>> import networkx
    >>> g = networkx.Graph()
    >>> g.add_nodes_from(sudoku.cells(boxsize))
    >>> g.add_edges_from(sudoku.dependent_cells(boxsize))

A Sudoku puzzle is a partial assignment of symbols, or colors, to the vertices of the graph. Graphs in ``networkx`` allow arbitrary data to be associated with graph nodes. ::

    >>> for cell in fixed:
    ...    g.node[cell]['color'] = fixed[cell]

A solution of the Sudoku puzzle is a minimal vertex coloring of the Sudoku graph which preserves the pre-assigned colors. There are many vertex coloring algorithms which we can use to try to find a solution of a puzzle. In ``sudoku.py`` we have implemented a general vertex coloring algorithm which can be customized to provide a variety of different algorithms. The effectiveness of these different algorithms is discussed in the later section about "Coloring the Sudoku graph"".

To use the graph model to find a solution to the Sudoku puzzle of Figure XXX, we can again call the ``solve`` function, specifying ``graph`` as the model. ::

    >>> s = sudoku.solve(d, 3, model = 'graph')

Polynomial system models
~~~~~~~~~~~~~~~~~~~~~~~~

The graph model above is mainly introduced in [Var05]_ as a prelude to modeling a Sudoku puzzle as a system of polynomial equations. The polynomial system model presented in [Var05]_ consists of a polynomial for every vertex in the graph model and a polynomial for every edge. 

Vertex polynomials have the form:

.. raw:: latex

   \[F(x_j) = \prod_{i=1}^{9} (x_j - i)\]

The Sudoku polynomial-system model in sudoku.py is implemented using ``sympy v0.6.7``. This open-source symbolic algebra Python library is available at `http://code.google.com/p/sympy/ <http://code.google.com/p/sympy/>`_

Vertex polynomials can be built using ``sympy`` as follows: ::

    >>> def node_polynomial(x, boxsize):
    ...    return reduce(operator.mul, [(x - symbol) for symbol in symbols(boxsize)])

Edge polynomials, for adjacent vertices :math:`$x_i$` and :math:`$x_j$`, have the form: 

.. raw:: latex

   \[G(x_i, x_j) = \frac{F(x_i) - F(x_j)}{x_i - x_j}\]

In ``sympy``, we build edge polynomials from the vertex polynomial function. ::

   >>> from sympy import cancel, expand
   >>> e = expand(cancel((node_polynomial(x, boxsize) - node_polynomial(y, boxsize))/(x - y)))

The polynomial model for the empty Sudoku puzzle now consists of the collection of all node polynomials for nodes in the Sudoku graph and all edge polynomials for node pairs ``(x,y) in dependent_symbols(boxsize)``, where the ``dependent_symbols`` function returns the list of dependent cells as pairs of ``sympy.Symobol`` objects.

To specify a Sudoku puzzle requires adding polynomials to represent the clues, or fixed cells. According to the model from [Var05]_, if :math:`$F$` is the set of fixed cells (i.e. cell label, value pairs) then to the polynomial system we need to add polynomials 
   
.. raw:: latex

   \[F(x_i, j) = x_i - j\]

Or, with ``sympy``: ::

    >>> sympy.Symbol('x' + str(cell)) - symbol

In Listing XXX, an example is shown of how to use the polynomial-system model to find a solution to the Sudoku puzzle of Figure XXX. ::

    >>> s = sudoku.solve(d, 3, model = 'groebner')
    >>> sudoku.print_puzzle(s, 3)

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

In ``pyglpk``, the integer program is represented as a matrix of coefficients of the linear equations which make up the integer program. Two functions of ``sudoku.py`` provide the correct dimensions of the coefficient matrix. ::

    >>> import glpk
    >>> lp = glpk.LPX()
    >>> lp.cols.add(lp_matrix_ncols(boxsize))
    >>> lp.rows.add(lp_matrix_nrows(boxsize))

Variables and equations are assigned bounds. ::

    >>> for c in lp.cols:
    ...    c.bounds = 0.0, 1.0
    >>> for r in lp.rows:
    ...    r.bounds = 1.0, 1.0

The linear program matrix. ::

    >>> lp.matrix = list(flatten(lp_coeffs(boxsize)))

Solving the linear relaxation first by the simplex algorithm :: 

    >>> lp.simplex()

and then using the ``glpk`` integer programming algorithm to solve the original integer programming problem: ::

    >>> for col in lp.cols:
    ...    col.kind = int
    >>> lp.integer()

Finally, we need to extract the solution as a dictionary: XXX THIS SHOULD REALLY BE A LITTLE FUNCTION CALL XXX ::
 
    >>> names = lp_vars(boxsize)
    >>> sol = {}
    >>> for c in lp.cols:
    ...    if c.value == 1:
    ...       sol[names[c.index][0]] = names[c.index][1]

In Listing XXX, an example is shown of how to use the integer programming model to find a solution to the Sudoku puzzle of Figure XXX. ::

    >>> s = sudoku.solve(d, 3, model = 'lp')
    >>> sudoku.print_puzzle(s, 3)
     2  5  8  7  3  6  9  4  1 
     6  1  9  8  2  4  3  5  7 
     4  3  7  9  1  5  2  6  8 
     3  9  5  2  7  1  4  8  6 
     7  6  2  4  9  8  1  3  5 
     8  4  1  6  5  3  7  2  9 
     1  8  4  3  6  9  5  7  2 
     5  7  6  1  4  2  8  9  3 
     9  2  3  5  8  7  6  1  4

Experimentation
---------------

In this section we demonstrate how to use XXX to create experimentation scripts. For the purposes of demonstration, we reproduce several results from the literature. We show how to enumerate Shidoku puzzles, how to color the Sudoku graph with the minimal number of colors, how to investigate minimally uniquely completable Sudoku puzzles, how to investigate phase transition phenomena in randomly generated Sudoku puzzles. Finally, we look at a competition, closely related to Sudoku puzzles, which was held by Mathworks in 2005. 

The intention of this section is to show how XXX makes the task of writing these experimental investigation scripts very easy.

Enumerating Shidoku
~~~~~~~~~~~~~~~~~~~

To solve the enumeration problem for Shidoku, using the constraint model implemented in `sudoku.py`, is straightforward. ::

    >>> setup_string = "from sudoku import empty_puzzle_as_CP"
    >>> experiment_string = """\
    ... p = empty_puzzle_as_CP(2)
    ... s = p.getSolutions()
    ... print len(s)"""
    >>> from timeit import Timer
    >>> t = Timer(experiment_string, setup_string)
    >>> print t.timeit(1)
    288
    0.146998882294

Coloring the Sudoku graph
~~~~~~~~~~~~~~~~~~~~~~~~~

As discussed above in the graph model section, a completed Sudoku puzzle is equivalent to a minimal vertex coloring of the Sudoku graph. 

We have experimented with several different vertex coloring algorithms to see which are more effective at coloring the Sudoku graph. At first, we used Joseph Culberson's graph coloring programs (available at XXXX) by writing the graph information to a file in Dimacs format (via the ``dimacs_string`` function of ``sudoku.py``). Observing the effectiveness of the saturation degree (DSatur) algorithm, we decided to implement a general verex coloring scheme directly in Python.

The vertex coloring algorithm of ``sudoku.py``, ``vertex_coloring``, has keyword parameters ``nodes`` and ``choose_color`` which allow for customization of the general scheme. The ``nodes`` parameter specifies an ordering of vertices which ``choose_color`` is a visitor object for selecting the color of an uncolored vertex.

For example, if ``nodes`` is assigned the ``InOrder`` class and ``choose_color`` the ``first_available_color`` function then ``vertex_coloring(graph, nodes, choose_color)`` is the basic sequential vertex coloring algorithm.

We can get a random order by assigning ``RandomOrder`` to ``nodes``.

The DSatur algorithm is obtained by choosing ``DSATOrder`` for ``nodes`` and ``first_available_color`` for ``choose_color``.

XXX DSatur algorithm is an online algorithm XXX supported because ``nodes`` is an iterator.

Minimal uniquely completable puzzles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gordon Royle maintains a list of uniquely completable 17-hint Sudoku puzzles at `http://mapleta.maths.uwa.edu.au/~gordon/sudoku17 <http://mapleta.maths.uwa.edu.au/~gordon/sudoku17>`_

XXX With ``sudoku.py`` we can write a short script for processing such puzzle collections XXX

Investigating hardness of random puzzles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

XXX A script for timing runs on random puzzles XXX

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

