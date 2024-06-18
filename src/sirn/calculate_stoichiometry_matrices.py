'''Calculates the reactant and product stoichiometry matrices.'''

import collections
import tellurium as te  # type: ignore
import simplesbml  # type: ignore
import numpy as np  # type: ignore

# Result: namedtuple
#  reactant_mat: reactant stoichiometry matrix
#  product_mat: product stoichiometry matrix
#  stoichiometry_mat: stoichiometry matrix
#  species_names: list of species names
#  reaction_names: list of reaction names
Result = collections.namedtuple('Result', 
        'reactant_mat, product_mat, stoichiometry_mat, species_names, reaction_names')


def CalculateStoichiometryMatrices(antimony_str:str)->Result:
    """
    Calculate the stoichiometry matrix from an SBML model.
    Args:
        antimony_str (str): Antimony model
    Returns:
        Result: Result(reactant_mat, product_mat, stoichiometry_mat, species_names, reaction_names)
    """
    roadrunner = te.loada(antimony_str)
    sbml = roadrunner.getSBML()
    model = simplesbml.loadSBMLStr(sbml)
    # Model inforeactant_mation
    num_species = model.getNumFloatingSpecies()
    num_reaction = model.getNumReactions()
    species_names = [model.getNthFloatingSpeciesId(i) for i in range(num_species)]
    reaction_names = [model.getNthReactionId(i) for i in range(num_reaction)]

    # Allocate space for the stoichiometry matrix
    reactant_mat = np.zeros((num_species, num_reaction))
    product_mat = np.zeros((num_species, num_reaction))
    stoichiometry_mat = np.zeros((num_species, num_reaction))
    for ispecies in range(num_species):
        floatingSpeciesId = model.getNthFloatingSpeciesId(ispecies)

        for ireaction in range(num_reaction):
            # Get the product stoihiometry
            numProducts = model.getNumProducts(ireaction)
            for k1 in range(numProducts):
                productId = model.getProduct(ireaction, k1)
                if floatingSpeciesId == productId:
                    product_mat[ispecies, ireaction] += model.getProductStoichiometry(ireaction, k1)
            # Get the reactant stoihiometry
            numReactants = model.getNumReactants(ireaction)
            for k1 in range(numReactants):
                reactantId = model.getReactant(ireaction, k1)
                reactionId = model.getNthReactionId(ireaction)
                if floatingSpeciesId == reactantId:
                    reactant_mat[ispecies, ireaction] += model.getReactantStoichiometry(ireaction, k1)
    # Calculate the stoichiometry matrix
    stoichiometry_mat = product_mat - reactant_mat
    return Result(reactant_mat=reactant_mat, product_mat=product_mat, stoichiometry_mat=stoichiometry_mat,
                  species_names=species_names, reaction_names=reaction_names)