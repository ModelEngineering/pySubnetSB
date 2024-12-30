'''Efficient container of properties for a reaction network.'''

from pySubnetSB import constants as cn  # type: ignore
from pySubnetSB.matrix import Matrix  # type: ignore
from pySubnetSB.named_matrix import NamedMatrix  # type: ignore
from pySubnetSB.stoichometry import Stoichiometry  # type: ignore
from pySubnetSB import util  # type: ignore
import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.assignment_pair import AssignmentPair  # type: ignore
from pySubnetSB.reaction_constraint import ReactionConstraint  # type: ignore
from pySubnetSB.species_constraint import SpeciesConstraint  # type: ignore

import collections
import json
import numpy as np
import os
import pandas as pd  # type: ignore
from pynauty import Graph  # type: ignore
import tellurium as te  # type: ignore
from typing import Optional, Tuple, Any, Union


Edge = collections.namedtuple('Edge', ['source', 'destination'])
GraphDescriptor = collections.namedtuple('GraphDescriptor', ['vertex_dct', 'label_dct'])
ConstraintPair = collections.namedtuple('ConstraintPair', ['reaction', 'species'])


class NetworkBase(object):
    """
    Abstraction for a reaction network. This is represented by reactant and product stoichiometry matrices.
    """

    def __init__(self, reactant_arr:Matrix, 
                 product_arr:np.ndarray,
                 reaction_names:Optional[np.ndarray[str]]=None,  # type: ignore
                 species_names:Optional[np.ndarray[str]]=None,   # type: ignore
                 network_name:Optional[str]=None)->None:
        """
        Args:
            reactant_arr (np.ndarray): Reactant matrix.
            product_arr (np.ndarray): Product matrix.
            network_name (str): Name of the network.
            reaction_names (np.ndarray[str]): Names of the reactions.
            species_names (np.ndarray[str]): Names of the species
        """
        self._network_name = network_name
        # Reactant stoichiometry matrix is negative
        if not np.all(reactant_arr.shape == product_arr.shape):
            raise ValueError("Reactant and product matrices must have the same shape.")
        try:
            self.num_species, self.num_reaction = np.shape(reactant_arr)
        except:
            # Empty network
            self.num_species = 0
            self.num_reaction = 0
            return
        if reactant_arr.shape[0] == 0:
            # Empty network
            self.num_species = 0
            self.num_reaction = 0
            return
        self.reactant_nmat = NamedMatrix(reactant_arr, row_names=species_names, column_names=reaction_names,
              row_description="species", column_description="reactions")
        self.product_nmat = NamedMatrix(product_arr, row_names=species_names, column_names=reaction_names,
              row_description="species", column_description="reactions")
        self.standard_nmat = NamedMatrix(product_arr - reactant_arr, row_names=species_names,
              column_names=reaction_names, row_description="species", column_description="reactions")
        # The following are deferred execution for efficiency considerations
        self._species_names = species_names
        self._reaction_names = reaction_names
        self._stoichiometry_nmat:Optional[NamedMatrix] = None
        self._network_hash:Optional[int] = None  # Hash for weak identity
        self._constraint_pair_dct:dict = {}  # keys are identity, is_subnet

    def __bool__(self)->bool:
        return (self.num_species > 0) and (self.num_reaction > 0)

    @property
    def stoichiometry_nmat(self)->NamedMatrix:
        if self._stoichiometry_nmat is None:
            self._stoichiometry_nmat = NamedMatrix(self.product_nmat.values - self.reactant_nmat.values,
                  row_names=self.species_names, column_names=self.reaction_names,
                  row_description="species", column_description="reactions")
        return self._stoichiometry_nmat

    @property
    def species_names(self)->np.ndarray[str]:  # type: ignore
        if self._species_names is None:
            self._species_names = np.array([f"S{i}" for i in range(self.num_species)])
        if not isinstance(self._species_names, np.ndarray):
            self._species_names = np.array(self._species_names)
        return self._species_names
    
    @property
    def reaction_names(self)->np.ndarray[str]:  # type: ignore
        if self._reaction_names is None:
            self._reaction_names = np.array([f"J{i}" for i in range(self.num_reaction)])
        if not isinstance(self._reaction_names, np.ndarray):
            self._reaction_names = np.array(self._reaction_names)
        return self._reaction_names

    @property
    def network_hash(self):
        if self._network_hash is None:
            self._network_hash = util.hashMatrix(self.standard_nmat.values)
        return self._network_hash

    @property
    def network_name(self)->str:
        if self._network_name is None:
            self._network_name = str(np.random.randint(0, 10000000))
        return self._network_name
    
    def resetNetworkName(self)->None:
        self._network_name = None

    def copy(self)->'NetworkBase':
        return NetworkBase(self.reactant_nmat.values.copy(), self.product_nmat.values.copy(),
                       network_name=self.network_name, reaction_names=self.reaction_names,
                       species_names=self.species_names)

    def __repr__(self)->str:
        repr = f"{self.network_name}: {self.num_species} species, {self.num_reaction} reactions"
        reactions = ["  " + self.prettyPrintReaction(i) for i in range(self.num_reaction)]
        repr += '\n' + '\n'.join(reactions)
        return repr
    
    def getConstraints(self, identity:str, is_subnet:bool)->ConstraintPair:
        """
        Get the constraints for the network.

        Args:
            identity (str): cn.ID_STRONG or cn.ID_WEAK
            is_subnet (bool): True if the constraints are for a subset.

        Returns:
            Tuple[SpeciesConstraint, ReactionConstraint]
        """
        key = (identity, is_subnet)
        if not key in self._constraint_pair_dct:
            reactant_nmat, product_nmat = self.makeMatricesForIdentity(identity)
            constraint_pair = ConstraintPair(
                  ReactionConstraint(reactant_nmat, product_nmat, is_subnet=is_subnet),
                  SpeciesConstraint(reactant_nmat, product_nmat, is_subnet=is_subnet))
            self._constraint_pair_dct[key] = constraint_pair
        return self._constraint_pair_dct[key]
    
    def makeMatricesForIdentity(self, identity:str)->Tuple[NamedMatrix, NamedMatrix]:
        """Calculates the reactant and product NamedMatrix based on the identity calculation being done.

        Args:
            identity (str): cn.ID_STRONG or cn.ID_WEAK
        """
        if identity == cn.ID_STRONG:
            return self.reactant_nmat, self.product_nmat
        else:
            reactant_arr = -1*(self.standard_nmat.values<0)*self.standard_nmat.values
            reactant_nmat = self.reactant_nmat.copy()
            reactant_nmat.values = reactant_arr
            product_arr = (self.standard_nmat.values>0)*self.standard_nmat.values
            product_nmat = self.product_nmat.copy()
            product_nmat.values = product_arr
            return reactant_nmat, product_nmat
        
    def isBoundaryNetwork(self)->bool:
        """
        A boundary network is one where all reactions are either synthesis or degradation of a single species.

        Args:
            network (Network): Network instance

        Returns:
            bool: True if the network has only one species
        """
        reactant_sum_arr = self.reactant_nmat.values.sum(axis=0)
        product_sum_arr = self.product_nmat.values.sum(axis=0)
        is_boundary = np.all((reactant_sum_arr + product_sum_arr) <= 1)
        return bool(is_boundary)
    
    def isMatrixEqual(self, other, identity:str=cn.ID_WEAK)->bool:
        """
        Check if the stoichiometry matrix is equal to another network's matrix.
            weak identity: standard stoichiometry matrix 
            strong identity: reactant and product matrices

        Args:
            other: Network
            identity (str, optional): Defaults to cn.ID_WEAK.

        Returns:
            bool
        """
        reference_reactant_nmat, reference_product_nmat = self.makeMatricesForIdentity(identity)
        target_reactant_nmat, target_product_nmat = other.makeMatricesForIdentity(identity)
        return bool(np.all(reference_reactant_nmat.values == target_reactant_nmat.values) 
            and np.all(reference_product_nmat.values == target_product_nmat.values))
    
    def __eq__(self, other)->bool:
        if self.network_name != other.network_name:
            return False
        return self.isEquivalent(other)
    
    def isEquivalent(self, other)->bool:
        """Same except for the network name.

        Args:
            other (_type_): _description_

        Returns:
            bool: _description_
        """
        if not isinstance(other, self.__class__):
            return False
        if not self.isMatrixEqual(other, identity=cn.ID_STRONG):
            return False
        if not np.all(self.species_names == other.species_names):
            return False
        if not np.all(self.reaction_names == other.reaction_names):
            return False
        return True
    
    def permute(self, assignment_pair:Optional[AssignmentPair]=None)->Tuple['NetworkBase', AssignmentPair]:
        """
        Creates a new network with permuted reactant and product matrices. If no permutation is specified,
        then a random permutation is used.

        Returns:
            BaseNetwork (class of caller)
            AssignmentPair (species_assignment, reaction_assignment) for reconstructing the original network.
        """
        #####
        def makePerm(size:int)->np.ndarray[int]:  # type: ignore
            # Creates a permutation of the desired legnth, ensuring that it's not the identity permutation
            identity = np.array(range(size))
            if size == 1:
                return identity
            for _ in range(100):
                perm = np.random.permutation(range(size))
                if not np.all(perm == identity):
                    break
            else:
                raise RuntimeError("Could not find a permutation.")
            return perm
            #####
        if assignment_pair is None:
            reaction_perm = makePerm(self.num_reaction)
            species_perm = makePerm(self.num_species)
        else:
            reaction_perm = assignment_pair.reaction_assignment
            species_perm = assignment_pair.species_assignment
        reactant_arr = self.reactant_nmat.values.copy()
        product_arr = self.product_nmat.values.copy()
        reactant_arr = reactant_arr[species_perm, :]
        reactant_arr = reactant_arr[:, reaction_perm]
        product_arr = product_arr[species_perm, :]
        product_arr = product_arr[:, reaction_perm]
        reaction_names = np.array(self.reaction_names[reaction_perm])
        species_names = np.array(self.species_names[species_perm])
        assignment_pair = AssignmentPair(np.argsort(species_perm), np.argsort(reaction_perm))
        return self.__class__(reactant_arr, product_arr,
              reaction_names=reaction_names, species_names=species_names), assignment_pair
    
    def isStructurallyCompatible(self, other:'NetworkBase', identity:str=cn.ID_WEAK)->bool:
        """
        Determines if two networks are compatible to be structurally identical.
        This means that they have the same species and reactions.

        Args:
            other (Network): Network to compare to.
            identity (str): cn.ID_WEAK or cn.ID_STRONG

        Returns:
            bool: True if compatible.
        """
        if self.num_species != other.num_species:
            return False
        if self.num_reaction != other.num_reaction:
            return False
        is_identity = self.network_hash == other.network_hash
        if identity == cn.ID_STRONG:
            is_identity = self.network_hash == other.network_hash
        return bool(is_identity)

    # FIXME: More sophisticated subset checking? 
    def isSubsetCompatible(self, other:'NetworkBase')->bool:
        """
        Determines if two networks are compatible in that self can be a subset of other.
        This means that they have the same species and reactions.

        Args:
            other (Network): Network to compare to.

        Returns:
            bool: True if compatible.
        """
        if self.num_species > other.num_species:
            return False
        if self.num_reaction > other.num_reaction:
            return False
        return True
    
    @classmethod
    def makeFromAntimonyStr(cls, antimony_str:str, network_name:Optional[str]=None,
          roadrunner:Optional[Any]=None)->Union['NetworkBase', None]:
        """
        Make a Network from an Antimony string.

        Args:
            antimony_str (str): Antimony string.
            network_name (str): Name of the network.
            roadrunner (Any): Roadrunner object.

        Returns:
            Network or None
        """
        if roadrunner is None:
            clean_antimony_str = antimony_str.replace("\n", ";\n")
        else:
            clean_antimony_str = None
        stoichiometry = Stoichiometry(clean_antimony_str, roadrunner=roadrunner)
        if stoichiometry.reactant_mat is None:
            return None
        network = cls(stoichiometry.reactant_mat, stoichiometry.product_mat, network_name=network_name,
                      species_names=stoichiometry.species_names, reaction_names=stoichiometry.reaction_names)
        return network
                   
    @classmethod
    def makeFromAntimonyFile(cls, antimony_path:str,
                         network_name:Optional[str]=None)->Union['NetworkBase', None]:
        """
        Make a Network from an Antimony file. The default network name is the file name.

        Args:
            antimony_path (str): path to an Antimony file.
            network_name (str): Name of the network.

        Returns:
            Network
        """
        with open(antimony_path, 'r') as fd:
            try:
                antimony_str = fd.read()
            except:
                raise ValueError(f"Could not read file {antimony_path}")
        if network_name is None:
            filename = os.path.basename(antimony_path)
            network_name = filename.split('.')[0]
        try:
            network = cls.makeFromAntimonyStr(antimony_str, network_name=network_name)
        except:
            network = None
        return network
    
    @classmethod
    def makeFromSBMLFile(cls, sbml_path:str,
                         network_name:Optional[str]=None)->Union['NetworkBase', None]:
        """
        Make a Network from an SBML file. The default network name is the file name.

        Args:
            sbml_path (str): path to an SBML file.
            network_name (str): Name of the network.

        Returns:
            Network
        """
        roadrunner = te.loadSBMLModel(sbml_path)
        stoichiometry = Stoichiometry(None, roadrunner=roadrunner)
        if stoichiometry.reactant_mat is None:
            return None
        network = cls(stoichiometry.reactant_mat, stoichiometry.product_mat, network_name=network_name,
                      species_names=stoichiometry.species_names, reaction_names=stoichiometry.reaction_names)
        return network
    
    @classmethod
    def makeRandomNetwork(cls, num_species:int=5, num_reaction:int=5)->'NetworkBase':
        """
        Makes a random network.

        Args:
            num_species (int): Number of species.
            num_reaction (int): Number of reactions.

        Returns:
            Network
        """
        reactant_arr = np.random.randint(0, 3, (num_species, num_reaction))
        product_arr = np.random.randint(0, 3, (num_species, num_reaction))
        return cls(reactant_arr, product_arr)
    
    @classmethod
    def makeRandomNetworkByReactionType(cls, num_reaction:int, num_species:int=-1, is_exact:bool=False,
              **kwargs)->'NetworkBase':
        """
        Makes a random network based on the type of reaction. Parameers are in the form
            <p<#products>r<#reactants>_frc> where #products and #reactants are the number of
            products and reactants. Ensures that reactions are unique.
        Fractions are from the paper "SBMLKinetics: a tool for annotation-independent classification of
            reaction kinetics for SBML models", Jin Liu, BMC Bioinformatics, 2023.

        Args:
            num_reaction (int): Number of reactions.
            num_species (int): Number of species. If <0, then num_species = num_reaction.
            is_exact (bool): Ensure that the number of species is exactly num_species.
            is_unique (bool): Ensure that reactions are unique.
            is_prune_species (bool): Prune species not used in any reaction.
            fractions by number of products and reactants

        Returns:
            Network
        """
        if num_reaction <= 0:
            raise ValueError("Number of reactions must be positive.")
        if not is_exact:
            return cls._makeRandomNetworkByReactionType(num_reaction, num_species=num_species, **kwargs)
        # Get the network consistent with the requirements
        max_attempts =  100000
        if num_species < 0:
            num_species = num_reaction
        for _ in range(max_attempts): 
            network = cls._makeRandomNetworkByReactionType(num_reaction, num_species=num_species, **kwargs)
            if network.num_species == num_species:
                return network
        raise ValueError(
              f"Could not find a network with {num_species} species and {num_reaction} reactions.")

    
    @classmethod
    def _makeRandomNetworkByReactionType(cls, 
              num_reaction:int,
              num_species:int=-1,
              is_prune_species:bool=True,
              p0r0_frc:Optional[float]=0.0,
              p0r1_frc:Optional[float]=0.1358,
              p0r2_frc:Optional[float]=0.001,
              p0r3_frc:Optional[float]=0.0,
              p1r0_frc:Optional[float]=0.0978,
              p1r1_frc:Optional[float]=0.3364,
              p1r2_frc:Optional[float]=0.1874,
              p1r3_frc:Optional[float]=0.0011,
              p2r0_frc:Optional[float]=0.0005,
              p2r1_frc:Optional[float]=0.1275,
              p2r2_frc:Optional[float]=0.0683,
              p2r3_frc:Optional[float]=0.0055,
              p3r0_frc:Optional[float]=0.0,
              p3r1_frc:Optional[float]=0.0087,
              p3r2_frc:Optional[float]=0.0154,
              p3r3_frc:Optional[float]=0.0146,
              is_unique:bool=False,
              )->'NetworkBase':
        """
        Makes a random network based on the type of reaction. Parameers are in the form
            <p<#products>r<#reactants>_frc> where #products and #reactants are the number of
            products and reactants. Ensures that reactions are unique.
        Fractions are from the paper "SBMLKinetics: a tool for annotation-independent classification of
            reaction kinetics for SBML models", Jin Liu, BMC Bioinformatics, 2023.

        Args:
            num_reaction (int): Number of reactions.
            num_species (int): Number of species.
            is_unique (bool): Ensure that reactions are unique.
            is_prune_species (bool): Prune species not used in any reaction.
            fractions by number of products and reactants

        Returns:
            Network
        """
        if num_species < 0:
            num_species = num_reaction
        SUFFIX = "_frc"
        # Initializations
        REACTION_TYPES = [f"p{i}r{j}" for i in range(4) for j in range(4)]
        FRAC_NAMES = [n + SUFFIX for n in REACTION_TYPES]
        value_dct:dict = {}
        total = 0
        for name in FRAC_NAMES:
            total += locals()[name] if locals()[name] is not None else 0
        for name in FRAC_NAMES:
            value = locals()[name]
            value_dct[name] = value/total
        CULMULATIVE_ARR = np.cumsum([value_dct[n + SUFFIX] for n in REACTION_TYPES])
        #######
        def getType(frac:float)->str:
            """
            Returns the name of the reaction associated with the fraction (e.g., a random (0, 1))

            Args:
                frac (float)

            Returns:
                str: Reaction type
            """
            pos = np.sum(CULMULATIVE_ARR < frac)
            for _ in range(len(REACTION_TYPES)):
                reaction_type = REACTION_TYPES[pos]
                # Handle cases of 0 fractions
                if not np.isclose(value_dct[reaction_type + SUFFIX], frac):
                    break
            return reaction_type
        #######
        MAX_ITERATION = 1000
        # Construct the reactions by building the reactant and product matrices
        reactant_arr = np.zeros((num_species, num_reaction))
        product_arr = np.zeros((num_species, num_reaction))
        i_reaction = 0
        num_iteration = 0
        while i_reaction < num_reaction:
            num_iteration += 1
            if num_iteration > max(num_reaction, MAX_ITERATION):
                raise ValueError("Could not find unique reactions.")
            frac = np.random.rand()
            reaction_type = getType(frac)
            num_product = int(reaction_type[1])
            num_reactant = int(reaction_type[3])
            # Products
            product_idxs = np.random.randint(0, num_species, num_product)
            product_arr[product_idxs, i_reaction] += 1
            # Reactants
            reactant_idxs = np.random.randint(0, num_species, num_reactant)
            reactant_arr[reactant_idxs, i_reaction] += 1
            # Check for unique reactions
            merged_arr = np.hstack([reactant_arr.T[:i_reaction+1], product_arr.T[:i_reaction+1]])
            if is_unique:
                uniques = np.unique(merged_arr, axis=0)
                if len(uniques) == i_reaction + 1:
                    # Unique reaction. Keep it
                    i_reaction += 1
                else:
                    # Duplicated reaction. Back it out
                    product_arr[product_idxs, i_reaction] -= 1
                    reactant_arr[reactant_idxs, i_reaction] -= 1
            else:
                i_reaction += 1
        # Eliminate 0 rows (species not used)
        if is_prune_species:
            keep_idxs:list = []
            for i_species in range(num_species):
                if np.sum(reactant_arr[i_species, :]) > 0 or np.sum(product_arr[i_species, :]) > 0:
                    keep_idxs.append(i_species)
            reactant_arr = reactant_arr[keep_idxs, :]
            product_arr = product_arr[keep_idxs, :]
        # Construct the network
        species_names = util.getDefaultSpeciesNames(reactant_arr.shape[0])
        reaction_names = util.getDefaultReactionNames(reactant_arr.shape[1])
        network = cls(reactant_arr, product_arr, reaction_names=reaction_names, species_names=species_names)
        return network
   
    def prettyPrintReaction(self, index:int)->str:
        """
        Pretty prints a reaction.

        Args:
            index (int): Index of the reaction.

        Returns:
            str
        """
        def makeSpeciesExpression(reaction_idx:int, stoichiometry_arr:np.ndarray)->str:
            all_idxs = np.array(range(self.num_species))
            species_idxs = all_idxs[stoichiometry_arr[:, reaction_idx] > 0]
            species_names = self.species_names[species_idxs]
            stoichiometries = [s for s in stoichiometry_arr[species_idxs, reaction_idx]]
            stoichiometries = ["" if np.isclose(s, 1) else str(s) + " " for s in stoichiometries]
            expressions = [f"{stoichiometries[i]}{species_names[i]}" for i in range(len(species_names))]
            result =  ' + '.join(expressions)
            return result
        #
        reactant_expression = makeSpeciesExpression(index, self.reactant_nmat.values)
        product_expression = makeSpeciesExpression(index, self.product_nmat.values)
        result = f"{self.reaction_names[index]}: " + f"{reactant_expression} -> {product_expression}"
        return result

    def makeNetworkFromAssignmentPair(self, assignment_pair:AssignmentPair)->'NetworkBase':
        """
        Constructs a network from an assignment pair.

        Args:
            assignment_pair (AssignmentPair): Assignment pair.

        Returns:
            Network: Network constructed from the assignment pair.
        """
        species_assignment = assignment_pair.species_assignment
        reaction_assignment = assignment_pair.reaction_assignment
        reactant_arr = self.reactant_nmat.values[species_assignment, :]
        product_arr = self.product_nmat.values[species_assignment, :]
        reactant_arr = reactant_arr[:, reaction_assignment]
        product_arr = product_arr[:, reaction_assignment]
        return NetworkBase(reactant_arr, product_arr, reaction_names=self.reaction_names[reaction_assignment],
                        species_names=self.species_names[species_assignment])
    
    def serialize(self)->str:
        """
        Serialize the network.

        Returns:
            str: string representation of json structure
        """
        reactant_lst = self.reactant_nmat.values.tolist()
        product_lst = self.product_nmat.values.tolist()
        dct = {cn.S_ID: self.__class__.__name__,
               cn.S_NETWORK_NAME: self.network_name,
               cn.S_REACTANT_LST: reactant_lst,
               cn.S_PRODUCT_LST: product_lst,
               cn.S_REACTION_NAMES: self.reaction_names.tolist(),
               cn.S_SPECIES_NAMES: self.species_names.tolist(),
               }
        return json.dumps(dct)

    @classmethod 
    def seriesToJson(cls, series:pd.Series)->str:
        """
        Convert a Series to a JSON serialization string.

        Args:
            ser: Series columns
                network_name, reactant_array_str, product_array_str, species_names, reaction_names,
                num_speces, um_reaction

        Returns:
            str: string representation of json structure
        """
        def convert(name):
            values = series[name]
            if isinstance(values, str):
                values = eval(values)
            return list(values)
        #####
        reactant_names = convert(cn.S_REACTION_NAMES)
        species_names = convert(cn.S_SPECIES_NAMES)
        num_reaction = len(reactant_names)
        num_species = len(species_names)
        product_arr = np.array(convert(cn.S_PRODUCT_LST))
        reactant_arr = np.array(convert(cn.S_REACTANT_LST))
        reactant_arr = np.reshape(reactant_arr, (num_species, num_reaction))
        product_arr = np.reshape(product_arr, (num_species, num_reaction))
        dct = {cn.S_ID: str(cls),
               cn.S_NETWORK_NAME: series[cn.S_NETWORK_NAME],
               cn.S_REACTANT_LST: reactant_arr.tolist(),
               cn.S_PRODUCT_LST: product_arr.tolist(),
               cn.S_REACTION_NAMES: reactant_names,
               cn.S_SPECIES_NAMES: species_names,
               }
        return json.dumps(dct)

    def toSeries(self)->pd.Series:
        """
        Serialize the network.

        Args:
            ser: Series columns
                network_name, reactant_array_str, product_array_str, species_names, reaction_names,
                num_speces, um_reaction

        Returns:
            str: string representation of json structure
        """
        dct = {cn.S_REACTANT_LST: self.reactant_nmat.values.flatten().tolist(),
               cn.S_PRODUCT_LST: self.product_nmat.values.flatten().tolist(),
               cn.S_NETWORK_NAME: self.network_name,
               cn.S_REACTION_NAMES: self.reaction_names,
               cn.S_SPECIES_NAMES: self.species_names}
        return pd.Series(dct)

    @classmethod 
    def deserialize(cls, serialization_str)->'NetworkBase':
        """
        Serialize the network.

        Returns:
            str: string representation of json structure
        """
        dct = json.loads(serialization_str)
        if not cls.__name__ in dct[cn.S_ID]:
            raise ValueError(f"Expected {cls} but got {dct[cn.S_ID]}")
        network_name = dct[cn.S_NETWORK_NAME]
        reactant_arr = np.array(dct[cn.S_REACTANT_LST])
        product_arr = np.array(dct[cn.S_PRODUCT_LST])
        reaction_names = np.array(dct[cn.S_REACTION_NAMES])
        species_names = np.array(dct[cn.S_SPECIES_NAMES])
        return cls(reactant_arr, product_arr, network_name=network_name,
                       reaction_names=reaction_names, species_names=species_names)
    
    def getGraphDescriptor(self, identity=cn.ID_STRONG)->GraphDescriptor:
        """
        Describes the bipartite graph of the network.
        Species are indices 0 to num_species - 1 and reactions are
        indices num_species to num_species + num_reaction - 1.
        Labels are provided to designate reaction and species nodes.

        Args:
            identity (str): cn.ID_STRONG or cn.ID_WEAK

        Returns:
            vertex_dict:
                key: source vertex index
                value: list of detination indices
            label_dict:
                key: vertex
                value: label ('reaction' or 'species')
        """
        num_species = self.num_species
        num_reaction = self.num_reaction
        num_node = num_species + num_reaction
        # Vertex dictionary
        vertex_dct:dict = {i: [] for i in range(num_node)}
        for i_reaction in range(num_reaction):
            for i_species in range(num_species):
                # Add an edge for each reactant
                species_vtx = i_species
                reaction_vtx = num_species + i_reaction
                if identity == cn.ID_STRONG:
                    # Reactants
                    for _ in range(int(self.reactant_nmat.values[i_species, i_reaction])):
                        vertex_dct[species_vtx].append(reaction_vtx)
                    # Products
                    for _ in range(int(self.product_nmat.values[i_species, i_reaction])):
                        vertex_dct[reaction_vtx].append(species_vtx)
                else:
                    # Weak identity. Use the standard stoichiometry matrix
                    stoichiometry = int(self.standard_nmat.values[i_species, i_reaction])
                    for _ in range(np.abs(stoichiometry)):
                        if stoichiometry > 0:
                            # Product
                            vertex_dct[reaction_vtx].append(species_vtx)
                        elif stoichiometry < 0:
                            # Reactant
                            vertex_dct[species_vtx].append(reaction_vtx)
                        else:
                            pass
        # Label dictionary
        label_dct:dict = {n: 'species' for n in range(num_species)}
        label_dct.update({n: 'reaction' for n in range(num_species, num_node)})

        return GraphDescriptor(vertex_dct=vertex_dct, label_dct=label_dct)

    def makePynautyNetwork(self, identity=cn.ID_STRONG)->Graph:
        """
        Make a pynauty graph from the network. Species are indices 0 to num_species - 1 and reactions are
        indices num_species to num_species + num_reaction - 1.

        Args:
            identity (str): cn.ID_STRONG or cn.ID_WEAK

        Returns:
            Graph: Pynauty graph
        """
        graph_descriptor = self.getGraphDescriptor(identity=identity)
        vertex_dct = graph_descriptor.vertex_dct
        graph = Graph(len(vertex_dct.keys()), directed=True)
        for node, neighbors in vertex_dct.items():
            graph.connect_vertex(node, neighbors)
        return graph
    
    def makeCSVNetwork(self, identity=cn.ID_STRONG)->str:
        """
        Creates a CSV representation of a directed graph. (See https://github.com/ciaranm/glasgow-subgraph-solver)
        indices num_species to num_species + num_reaction - 1.
        The CSV format has one line for each edge.  j
          <source>> <destination>

        Args:
            identity (str): cn.ID_STRONG or cn.ID_WEAK

        Returns:
            Graph: Pynauty graph
        """
        graph_descriptor = self.getGraphDescriptor(identity=identity)
        outputs = []
        # Create the vertices
        for source, destinations in graph_descriptor.vertex_dct.items():
            for destination in destinations:
                outputs.append(f"{source}>{destination}")
        # Add the labels
        for vertex, label in graph_descriptor.label_dct.items():
            outputs.append(f"{vertex},,{label}")
        return '\n'.join(outputs)
    
    def fill(self, num_fill_species:int=3, num_fill_reaction:Optional[int]=3,
               is_permute:bool=True)->'NetworkBase':
        """Creates a new network that augments (fills) the existing network with new species and reactions.

        Args:
            num_fill_species (int): Number of filler species. Use num_reaction and num_species if None.
            num_fill_reaction (int): Number of filler reactions. Use num_species and num_reaction if None.
            is_permute (bool): Permute the network.

        Returns:
            Network
        """
        # Check options
        if num_fill_species is None and num_fill_reaction is None:
            raise ValueError("Must specify either num_fill_species or num_fill_reaction.")
        if num_fill_species is None:
            num_fill_species = num_fill_reaction
        if num_fill_reaction is None:
            num_fill_reaction = num_fill_species
        # Create the filler network
        if num_fill_species < 1:  # type: ignore
            raise ValueError("Number of filler species must be at least 1.")
        if num_fill_reaction < 1: # type: ignore
            raise ValueError("Number of filler reaction must be at least 1.")
        filler_network = NetworkBase.makeRandomNetworkByReactionType(num_fill_reaction,  # type: ignore
            num_species=num_fill_species)
        # Creates a supernetwork with the reference in the upper left corner of the matrices
        # and the filler network in the bottom right. Then, randomize.
        self_pad_arr = np.zeros((self.num_species, filler_network.num_reaction))   # type: ignore
        filler_pad_arr = np.zeros((filler_network.num_species, self.num_reaction)) # type: ignore
        #####
        def makeTargetArray(reference_arr:np.ndarray, filler_arr:np.ndarray)->np.ndarray:
            padded_reference_arr = np.hstack([reference_arr.copy(), self_pad_arr])
            padded_filler_arr = np.hstack([filler_pad_arr, filler_arr.copy()])
            target_arr = np.vstack([padded_reference_arr, padded_filler_arr])
            return target_arr
        ##### 
        # Construct the reactant array so that reference is upper left and filler is lower right
        target_reactant_arr = makeTargetArray(self.reactant_nmat.values,
              filler_network.reactant_nmat.values)
        target_product_arr = makeTargetArray(self.product_nmat.values,
              filler_network.product_nmat.values)
        target_network = self.__class__(target_reactant_arr, target_product_arr)
        if is_permute:
            target_network, _ = target_network.permute()
        return target_network
    
    def makeInducedNetwork(self, assignment_pair:AssignmentPair)->'NetworkBase':
        """
        Makes an induced network based on an assignment pair.

        Args:
            assignment_pair (AssignmentPair): Assignment pair.

        Returns:
            Network
        """
        species_assignment = assignment_pair.species_assignment
        reaction_assignment = assignment_pair.reaction_assignment
        reactant_arr = self.reactant_nmat.values[species_assignment, :]
        product_arr = self.product_nmat.values[species_assignment, :]
        reactant_arr = reactant_arr[:, reaction_assignment]
        product_arr = product_arr[:, reaction_assignment]
        return self.__class__(reactant_arr, product_arr, reaction_names=self.reaction_names[reaction_assignment],
                        species_names=self.species_names[species_assignment])