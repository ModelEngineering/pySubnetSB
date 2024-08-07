"""
Key structures in DSIRN

A Matrix is a two dimensional numpy array.
A StoichiometryMatrix is a Matrix whose rows are species and columns are reactions.
A Vector is a one dimensional numpy array.
A Criterion is a boolean valued function of a real number.
A CriterionVector is a Vector of Criteria.
A CriteriaCountMatrix for a Matrix M and a CriteriaVector c is a Matrix is i,j entery is
    the counts of values in the i-th row of M that satisfy the j-th criterion in c.
An assignment of Matrix T to Matrix R is a sequence of indices of rows in T such that: (a) the length of
    the sequence is the number of rows in R; and (b) there is no repetition of rows in the sequence.
A CriteriaPairMatrixCount for a Matrix M and CriteriaVector c has rows indexed by a pair of indices i,j
    columns indexed by a pair of indices k,l. The element in row (i,j) and column (k,l) is the number of
    columns n such that M[i,n] statisfies c[k] and M[j,n] satisfies c[l].
"""