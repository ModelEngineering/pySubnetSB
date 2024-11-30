"""
Key structures in pySubnetSB

A NamedMatrix is a two dimensional numpy array with row and column names.
A Network is a representation of a chemical reaction network that includes a NamedMatrix for reactants and products.
A Constraint describes numerical or categorical charateristics of a network that are organized as one or more NamedMatrix.
"""
import sirn.constants as cn  # type: ignore


def findForDirectories(*args, **kwargs):
    from sirn.api import findForDirectories
    return findForDirectories(*args, **kwargs)

def findForModels(*args, **kwargs):
    from sirn.api import findForModels
    return findForModels(*args, **kwargs)

def serialize(*args, **kwargs):
    from sirn.api import serialize
    return serialize(*args, **kwargs)