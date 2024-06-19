# DETECTING STRUCTURALLY IDENTICAL REACTION NETWORKS

Two **chemical reaction networks (CRNs)** are structurally identical if they have the same stoichiometry for reactants (reactant stoichiometry matrix) and products (product stoichiometry matrix), regardless of their rate laws. Because of renamings of chemical species and reactions, testing for structurally
identical networks requires an element-wise comparison of every permutation of the rows and columns of the
stoichiometry matrices. That is, if two networks are structurally identical then they have permutably identical stoichiometry reactant matrices and their product stoichiometry matrix are identical for the same permutations used to get equal reactant matrices. Clearly, this definition applies if we exchange "reactant" and "product".

Why do we use the above definition and not simply the stoichiometry matrix? This is best answered by showing an example. Consider the following two networks that consist of a single reaction:

    // Network 1
    S1 -> S1 + S2

    // Network 2
    S2 -> S2 + S2

These networks have the same Stoichiometry matrix that has a 0 for ``S1`` and a 1 for ``S2``. However, the reactant and product stoichiometry matrices are different for these two networks.

# Problem Addressed
The above approach to finding structurally identical CRNs has a huge computational complexity.
Let $N$ be the number of rows (species) in a stoichiometry matrix and M be the number of columns.
Then, the computational complexity of a single pair-wise comparison is $O(N!M!)$. If each comparison takes 1 microsecond, then: (i) $N=8=M$ takes about an hour; (ii) N=10=M takes about a day; and (iii) $N=20=M$ takes
longer than the current age of the Universe (14B years). In systems biology, $N=20=M$ is a modest size CRN.

# Technical Approach
This project implements the DSIRN Algorithm, an efficient algorithm for detecting
structurally identical CRNs.
The key insight used by the algorithm is to eliminate the need for considering a large number of permutations.
This is achieved by finding an **order independent encoding (OIE)** of rows and columns of the stoichiometry matrix
so that rows (columns) are only compared if they have the same OIE. The stoichiometry matrix of many CRNs is dominated by -1, 0, 1 because of the wide prevalence of unit stoichiometries. So, we use the OIE
* number of elements < 0
* number of elements = 0
* number of elements > 0

By so doing, we partition the rows (columns) so that we only need to consider the permutations in each partition.
Let $N_P$ be the number of rows of with distinct OIE encodings for two structurally identical matrices, and $M_P$ be the same for the number of columns.
We can approximate the speedup by assuming that each partition of rows has the same number of elements, and a similar approximation can be made for the columns. Under these assumpations, the speedup of our algorithm is
$(N_P)^N \times (M_P)^M$.

The approximation works well for a smaller number of partitions. Consider the extreme case that $N_P = 1 = M_P$. And so,
$(N_P)^N \times (M_P)^M = 1^N \times 1^M = 1.$ That is, there is no speedup. The approximation at the other extreme is less accurate. That is, if $N_P = N$, then there is only one ordering of the rows consistent with the OIE. Similarly, for $M_P = M$ for columns. Thus, we estimate the total number of permutations to be $N^N \times M^M >> N!M!.$

# Design

* A **``Matrix``** is a two dimensional numpy array.
* An **``ArrayCollection``** is a collection of one dimensional arrays of the same length. An ``ArrayCollection`` knows the OIEs of its arrays.
* A **``PMatrix``** is ``Matrix`` that provides an efficient test for permutably identical matrices. ``PMatrix`` has an ``ArrayCollection`` for it rows and another for its columns. ``PMatrix`` uses ``ArrayCollection`` to obtain OIEs from which it constructs partitions of rows (columns) to do efficient checking for permutably identical matrices.
* A **``Network``** represents a CRN. It has a reactant ``PMatrix`` that represents the reactant stoichiometry matrix, and a product ``PMatrix`` that represents the product stoichiometry matrix. ``Network`` has a hash that is calculated from the reactant and product ``PMatrix``.
* and product stoichiometry matrices.
* A **NetworkCollection** is a collection of Network. NetworkCollection provides a way to discover subsets that are structurally identical. ``NetworkCollection.is_structurally_identical`` indicates if all Network in the collection are structurally identical.
* A **NCSerializer** provides a way to save and restore a NetworkCollection. It also provides for creating a NetworkCollection from a directory of Antimony Files.