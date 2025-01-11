'''Calculates the reactant, product, net stoichiometry matrices.'''
# Originally developed by Herbert M Sauro, University of Washington.

import tellurium as te  # type: ignore
import simplesbml  # type: ignore
import numpy as np  # type: ignore
from typing import Any, Optional

# Result: namedtuple
#  reactant_mat: reactant stoichiometry matrix
#  product_mat: product stoichiometry matrix
#  standard_mat: the usual stoichiometry matrix
#  species_names: list of species names
#  reaction_names: list of reaction names


class Stoichiometry(object):

    def __init__(self, antimony_str:str, roadrunner:Optional[Any]=None)->None:

        """
        Calculate the stoichiometry matrix from an SBML model.
        Args:
            antimony_str (str): Antimony model
        Returns:
            Result: Result(reactant_mat, product_mat, stoichiometry_mat, species_names, reaction_names)
        """
        self.antimony_str = antimony_str
        self.roadrunner = roadrunner
        self.reactant_mat, self.product_mat, self.standard_mat, self.species_names, self.reaction_names \
            = self.calculate() 
        
    def calculate(self):
        if self.roadrunner is None:
            self.roadrunner = te.loada(self.antimony_str)
        #roadrunner.conservedMoietyAnalysis = True
        sbml = self.roadrunner.getSBML()
        try:
            model = simplesbml.loadSBMLStr(sbml)
        except:
            return None, None, None, None, None
        # Model inforeactant_mation
        num_floating_species = model.getNumFloatingSpecies()
        num_boundary_species = model.getNumBoundarySpecies()
        num_species = num_floating_species + num_boundary_species
        num_reaction = model.getNumReactions()
        species_names = [model.getNthFloatingSpeciesId(i) for i in range(num_floating_species)]
        species_names.extend([model.getNthBoundarySpeciesId(i) for i in range(num_boundary_species)])
        reaction_names = [model.getNthReactionId(i) for i in range(num_reaction)]

        # Allocate space for the stoichiometry matrix
        reactant_mat = np.zeros((num_species, num_reaction))
        product_mat = np.zeros((num_species, num_reaction))
        standard_mat = np.zeros((num_species, num_reaction))
        for ispecies in range(num_species):
            if ispecies < num_floating_species:
                speciesId = model.getNthFloatingSpeciesId(ispecies)
            else:
                speciesId = model.getNthBoundarySpeciesId(ispecies - num_floating_species)
            for ireaction in range(num_reaction):
                # Get the product stoichiometry
                numProducts = model.getNumProducts(ireaction)
                for k1 in range(numProducts):
                    productId = model.getProduct(ireaction, k1)
                    if speciesId == productId:
                        product_mat[ispecies, ireaction] += model.getProductStoichiometry(ireaction, k1)
                # Get the reactant stoihiometry
                numReactants = model.getNumReactants(ireaction)
                for k1 in range(numReactants):
                    reactantId = model.getReactant(ireaction, k1)
                    #reactionId = model.getNthReactionId(ireaction)
                    if speciesId == reactantId:
                        reactant_mat[ispecies, ireaction] += model.getReactantStoichiometry(ireaction, k1)
        # Calculate the stoichiometry matrix
        standard_mat = product_mat - reactant_mat
        num_row, num_column = standard_mat.shape
        if (num_row != len(species_names)) or (num_column != len(reaction_names)):
            raise RuntimeError("The stoichiometry matrix is not the correct size!")
        # return result
        return reactant_mat, product_mat, standard_mat, species_names, reaction_names
