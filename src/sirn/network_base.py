'''Efficient container of properties for a reaction network.'''

from sirn import constants as cn  # type: ignore
from sirn.criteria_vector import CriteriaVector  # type: ignore
from sirn.matrix import Matrix  # type: ignore
from sirn.named_matrix import NamedMatrix  # type: ignore
from sirn.pair_criteria_count_matrix import PairCriteriaCountMatrix  # type: ignore
from sirn.single_criteria_count_matrix import SingleCriteriaCountMatrix  # type: ignore
from sirn.stoichometry import Stoichiometry  # type: ignore
from sirn import util  # type: ignore
import sirn.constants as cn # type: ignore
from sirn.assignment_pair import AssignmentPair  # type: ignore

import collections
import itertools
import json
import os
import pandas as pd  # type: ignore
from pynauty import Graph  # type: ignore
import numpy as np
from typing import Optional, Tuple, List, Dict


CRITERIA_VECTOR = CriteriaVector()
Edge = collections.namedtuple('Edge', ['source', 'destination'])


class NetworkBase(object):
    """
    Abstraction for a reaction network. This is represented by reactant and product stoichiometry matrices.
    """

    def __init__(self, reactant_arr:Matrix, 
                 product_arr:np.ndarray,
                 reaction_names:Optional[np.ndarray[str]]=None,
                 species_names:Optional[np.ndarray[str]]=None,
                 network_name:Optional[str]=None,
                 criteria_vector:CriteriaVector=CRITERIA_VECTOR)->None:
        """
        Args:
            reactant_mat (np.ndarray): Reactant matrix.
            product_mat (np.ndarray): Product matrix.
            network_name (str): Name of the network.
            reaction_names (np.ndarray[str]): Names of the reactions.
            species_names (np.ndarray[str]): Names of the species
        """
        # Reactant stoichiometry matrix is negative
        if not np.all(reactant_arr.shape == product_arr.shape):
            raise ValueError("Reactant and product matrices must have the same shape.")
        self.num_species, self.num_reaction = np.shape(reactant_arr)
        self.criteria_vector = criteria_vector
        self.reactant_mat = NamedMatrix(reactant_arr, row_names=species_names, column_names=reaction_names,
              row_description="species", column_description="reactions")
        self.product_mat = NamedMatrix(product_arr, row_names=species_names, column_names=reaction_names,
              row_description="species", column_description="reactions")
        self.standard_mat = NamedMatrix(product_arr - reactant_arr, row_names=species_names,
              column_names=reaction_names, row_description="species", column_description="reactions")
        # The following are deferred execution for efficiency considerations
        self._species_names = species_names
        self._reaction_names = reaction_names
        self._network_name = network_name
        self._stoichiometry_mat:Optional[NamedMatrix] = None
        self._network_mats:Optional[dict] = None # Network matrices populated on demand by getNetworkMat
        self._strong_hash:Optional[int] = None  # Hash for strong identity
        self._weak_hash:Optional[int] = None  # Hash for weak identity

    # Properties for handling deferred execution
    @property
    def species_names(self)->np.ndarray[str]:
        if self._species_names is None:
            self._species_names = np.array([f"S{i}" for i in range(self.num_species)])
        if not isinstance(self._species_names, np.ndarray):
            self._species_names = np.array(self._species_names)
        return self._species_names
    
    @property
    def reaction_names(self)->np.ndarray[str]:
        if self._reaction_names is None:
            self._reaction_names = np.array([f"J{i}" for i in range(self.num_reaction)])
        if not isinstance(self._reaction_names, np.ndarray):
            self._reaction_names = np.array(self._reaction_names)
        return self._reaction_names

    @property
    def weak_hash(self):
        if self._weak_hash is None:
            single_criteria_matrices = [self.getNetworkMatrix(
                                  matrix_type=cn.MT_SINGLE_CRITERIA, orientation=o,
                                  identity=cn.ID_WEAK)
                                  for o in [cn.OR_SPECIES, cn.OR_REACTION]]
            # Maintain the order of species and reactions
            hash_arr = np.sort([s.row_order_independent_hash for s in single_criteria_matrices])
            self._weak_hash = util.makeRowOrderIndependentHash(hash_arr)
        return self._weak_hash
        
    @property
    def strong_hash(self):
        if self._strong_hash is None:
            orientations = [cn.OR_SPECIES, cn.OR_REACTION]
            participants = [cn.PR_REACTANT, cn.PR_PRODUCT]
            combinations = itertools.product(orientations, participants)
            single_criteria_matrices = [self.getNetworkMatrix(
                  matrix_type=cn.MT_SINGLE_CRITERIA, orientation=o, identity=cn.ID_STRONG, participant=p)
                  for o, p in combinations]
            hash_arr = np.sort([s.row_order_independent_hash for s in single_criteria_matrices])
            self._strong_hash = util.makeRowOrderIndependentHash(hash_arr)
        return self._strong_hash
        
    @property
    def deprecated_strong_hash(self)->int:
        if self._strong_hash is None:
            stoichiometries:list = []
            for i_orientation in cn.OR_LST:
                for i_participant in cn.PR_LST:
                    stoichiometries.append(self.getNetworkMatrix(
                        matrix_type=cn.MT_SINGLE_CRITERIA,
                        orientation=i_orientation,
                        identity=cn.ID_STRONG,
                        participant=i_participant))
            #hash_arr = np.array([hashArray(stoichiometry.row_hashes.values) for stoichiometry in stoichiometries])
            #hash_arr = np.sort(hash_arr)
            #self._strong_hash = hashArray(hash_arr)
            self._strong_hash = hash(str(stoichiometries))
        return self._strong_hash

    @property
    def network_name(self)->str:
        if self._network_name is None:
            self._network_name = str(np.random.randint(0, 10000000))
        return self._network_name
    
    def resetNetworkName(self)->None:
        self._network_name = None

    @property
    def stoichiometry_mat(self)->NamedMatrix:
        if self._stoichiometry_mat is None:
            stoichiometry_arr = self.product_mat.values - self.reactant_mat.values
            self._stoichiometry_mat = NamedMatrix(stoichiometry_arr, row_names=self.species_names,
               column_names=self.reaction_names, row_description="species", column_description="reactions")
        return self._stoichiometry_mat

    # Methods 
    def getNetworkMatrix(self,
                         matrix_type:Optional[str]=None,
                         orientation:Optional[str]=None,
                         participant:Optional[str]=None,
                         identity:Optional[str]=None)->NamedMatrix: # type: ignore
        """
        Retrieves, possibly constructing, the matrix. The specific matrix is determined by the arguments.

        Args:
            marix_type: cn.MT_STOICHIOMETRY, cn.MT_SINGLE_CRITERIA, cn.MT_PAIR_CRITERIA
            orientation: cn.OR_REACTION, cn.OR_SPECIES
            participant: cn.PR_REACTANT, cn.PR_PRODUCT
            identity: cn.ID_WEAK, cn.ID_STRONG

        Returns:
            subclass of Matrix
        """
        # Initialize the dictionary of matrices
        if self._network_mats is None:
            self._network_mats = {}
            for i_matrix_type in cn.MT_LST:
                for i_orientation in cn.OR_LST:
                    for i_identity in cn.ID_LST:
                        for i_participant in cn.PR_LST:
                            if i_identity == cn.ID_WEAK:
                                self._network_mats[(i_matrix_type, i_orientation, None, i_identity)] = None
                            else:
                                self._network_mats[(i_matrix_type, i_orientation, i_participant, i_identity)] = None
        # Check if the matrix is already in the dictionary
        if self._network_mats[(matrix_type, orientation, participant, identity)] is not None:
            return self._network_mats[(matrix_type, orientation, participant, identity)]
        # Obtain the matrix value
        #   Identity and participant
        if identity == cn.ID_WEAK:
            matrix = self.stoichiometry_mat
        elif identity == cn.ID_STRONG:
            if participant == cn.PR_REACTANT:
                matrix = self.reactant_mat
            elif participant == cn.PR_PRODUCT:
                matrix = self.product_mat
            else:
                raise ValueError("Invalid participant: {participant}.")
        else:
            raise ValueError("Invalid identity: {identity}.")
        #   Orientation
        if orientation == cn.OR_REACTION:
            matrix = matrix.transpose()
        elif orientation == cn.OR_SPECIES:
            pass
        else:
            raise ValueError("Invalid orientation: {orientation}.")
        #   Matrix type
        if matrix_type == cn.MT_SINGLE_CRITERIA:
            matrix = SingleCriteriaCountMatrix(matrix.values, criteria_vector=self.criteria_vector)
        elif matrix_type == cn.MT_PAIR_CRITERIA:
            matrix = PairCriteriaCountMatrix(matrix.values, criteria_vector=self.criteria_vector)
        elif matrix_type == cn.MT_STOICHIOMETRY:
            pass
        else:
            raise ValueError("Invalid matrix type: {matrix_type}.")
        # Update the matrix
        self._network_mats[(matrix_type, orientation, participant, identity)] = matrix
        return matrix

    def copy(self)->'NetworkBase':
        return NetworkBase(self.reactant_mat.values.copy(), self.product_mat.values.copy(),
                       network_name=self.network_name, reaction_names=self.reaction_names,
                       species_names=self.species_names,
                       criteria_vector=self.criteria_vector)  # type: ignore

    def __repr__(self)->str:
        repr = f"{self.network_name}: {self.num_species} species, {self.num_reaction} reactions"
        reactions = ["  " + self.prettyPrintReaction(i) for i in range(self.num_reaction)]
        repr += '\n' + '\n'.join(reactions)
        return repr
    
    def isMatrixEqual(self, other, identity:str=cn.ID_WEAK)->bool:
        """
        Check if the stoichiometry matrix is equal to another network's matrix.
            weak identity: standard stoichiometry matrix 
            strong identity: reactant and product matrices

        Args:
            other (_type_): Network
            identity (str, optional): Defaults to cn.ID_WEAK.

        Returns:
            bool
        """
        def check(matrix_type, identity, participant=None):
            matrix1 = self.getNetworkMatrix(matrix_type=matrix_type, orientation=cn.OR_SPECIES,
                participant=participant, identity=identity)
            matrix2 = other.getNetworkMatrix(matrix_type=matrix_type, orientation=cn.OR_SPECIES,
                participant=participant, identity=identity)
            if not np.all(matrix1.shape == matrix2.shape):
                return False
            return np.all(matrix1.values == matrix2.values)
        #
        if identity == cn.ID_WEAK:
            if not check(cn.MT_STOICHIOMETRY, identity):
                return False
        else:
            if not check(cn.MT_STOICHIOMETRY, identity, participant=cn.PR_REACTANT):
                return False
            if not check(cn.MT_STOICHIOMETRY, identity, participant=cn.PR_PRODUCT):
                return False
        return True
    
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
        def makePerm(size:int)->np.ndarray[int]:
            # Creates a permutation of the desired legnth, ensuring that it's not the identity permutation
            identity = np.array(range(size))   
            for _ in range(10):
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
        reactant_arr = self.reactant_mat.values.copy()
        product_arr = self.product_mat.values.copy()
        reactant_arr = reactant_arr[species_perm, :]
        reactant_arr = reactant_arr[:, reaction_perm]
        product_arr = product_arr[species_perm, :]
        product_arr = product_arr[:, reaction_perm]
        reaction_names = np.array([self.reaction_names[i] for i in reaction_perm])
        species_names = np.array([self.species_names[i] for i in species_perm])
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
        is_identity = self.weak_hash == other.weak_hash
        if identity == cn.ID_STRONG:
            is_identity = self.strong_hash == other.strong_hash
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
    def makeFromAntimonyStr(cls, antimony_str:str, network_name:Optional[str]=None)->'NetworkBase':
        """
        Make a Network from an Antimony string.

        Args:
            antimony_str (str): Antimony string.
            network_name (str): Name of the network.

        Returns:
            Network
        """
        stoichiometry = Stoichiometry(antimony_str)
        network = cls(stoichiometry.reactant_mat, stoichiometry.product_mat, network_name=network_name,
                      species_names=stoichiometry.species_names, reaction_names=stoichiometry.reaction_names)
        return network
                   
    @classmethod
    def makeFromAntimonyFile(cls, antimony_path:str,
                         network_name:Optional[str]=None)->'NetworkBase':
        """
        Make a Network from an Antimony file. The default network name is the file name.

        Args:
            antimony_path (str): path to an Antimony file.
            network_name (str): Name of the network.

        Returns:
            Network
        """
        with open(antimony_path, 'r') as fd:
            antimony_str = fd.read()
        if network_name is None:
            filename = os.path.basename(antimony_path)
            network_name = filename.split('.')[0]
        return cls.makeFromAntimonyStr(antimony_str, network_name=network_name)
    
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
        reactant_mat = np.random.randint(0, 3, (num_species, num_reaction))
        product_mat = np.random.randint(0, 3, (num_species, num_reaction))
        return cls(reactant_mat, product_mat)
    
    @classmethod
    def makeRandomNetworkByReactionType(cls, 
              num_reaction:int, 
              num_species:Optional[int]=None,
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
              )->'NetworkBase':
        """
        Makes a random network based on the type of reaction. Parameers are in the form
            <p<#products>r<#reactants>_frc> where #products and #reactants are the number of
            products and reactants
        Fractions are from the paper "SBMLKinetics: a tool for annotation-independent classification of
            reaction kinetics for SBML models", Jin Liu, BMC Bioinformatics, 2023.

        Args:
            num_reaction (int): Number of reactions.
            num_species (int): Number of species.
            is_prune_species (bool): Prune species not used in any reaction.
            fractions by number of products and reactants

        Returns:
            Network
        """
        SUFFIX = "_frc"
        # Handle defaults
        if num_species is None:
            num_species = num_reaction
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
        # Initialize the reactant and product matrices
        reactant_arr = np.zeros((num_species, num_reaction))
        product_arr = np.zeros((num_species, num_reaction))
        # Construct the reactions by building the reactant and product matrices
        for i_reaction in range(num_reaction):
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
        # Eliminate 0 rows (species not used)
        if is_prune_species:
            keep_idxs:list = []
            for i_species in range(num_species):
                if np.sum(reactant_arr[i_species, :]) > 0 or np.sum(product_arr[i_species, :]) > 0:
                    keep_idxs.append(i_species)
            reactant_arr = reactant_arr[keep_idxs, :]
            product_arr = product_arr[keep_idxs, :]
        # Construct the network
        network = cls(reactant_arr, product_arr)
        return network
   
    def prettyPrintReaction(self, index:int)->str:
        """
        Pretty prints a reaction.

        Args:
            index (int): Index of the reaction.

        Returns:
            str
        """
        def makeSpeciesExpression(reaction_idx:int, stoichiometry_mat:np.ndarray)->str:
            all_idxs = np.array(range(self.num_species))
            species_idxs = all_idxs[stoichiometry_mat[:, reaction_idx] > 0]
            species_names = self.species_names[species_idxs]
            stoichiometries = [s for s in stoichiometry_mat[species_idxs, reaction_idx]]
            stoichiometries = ["" if np.isclose(s, 1) else str(s) + " " for s in stoichiometries]
            expressions = [f"{stoichiometries[i]}{species_names[i]}" for i in range(len(species_names))]
            result =  ' + '.join(expressions)
            return result
        #
        reactant_expression = makeSpeciesExpression(index, self.reactant_mat.values)
        product_expression = makeSpeciesExpression(index, self.product_mat.values)
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
        reactant_arr = self.reactant_mat.values[species_assignment, :]
        product_arr = self.product_mat.values[species_assignment, :]
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
        reactant_lst = self.reactant_mat.values.tolist()
        product_lst = self.product_mat.values.tolist()
        criteria_vector_serialization = self.criteria_vector.serialize()
        dct = {cn.S_ID: self.__class__.__name__,
               cn.S_NETWORK_NAME: self.network_name,
               cn.S_REACTANT_LST: reactant_lst,
               cn.S_PRODUCT_LST: product_lst,
               cn.S_REACTION_NAMES: self.reaction_names.tolist(),
               cn.S_SPECIES_NAMES: self.species_names.tolist(),
               cn.S_CRITERIA_VECTOR: criteria_vector_serialization,
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
        dct = {cn.S_REACTANT_LST: self.reactant_mat.values.flatten().tolist(),
               cn.S_PRODUCT_LST: self.product_mat.values.flatten().tolist(),
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
        if cn.S_CRITERIA_VECTOR in dct.keys():
            criteria_vector = CriteriaVector.deserialize(dct[cn.S_CRITERIA_VECTOR])
        else:
            criteria_vector = None
        return cls(reactant_arr, product_arr, network_name=network_name,
                       reaction_names=reaction_names, species_names=species_names,
                       criteria_vector=criteria_vector)
    
    def getGraphDct(self, identity=cn.ID_STRONG)->Dict[int, List[int]]:
        """
        Describes the bipartite graph of the network.
        Species are indices 0 to num_species - 1 and reactions are
        indices num_species to num_species + num_reaction - 1.

        Args:
            identity (str): cn.ID_STRONG or cn.ID_WEAK

        Returns:
            dict:
                key: source index
                value: list of detination index
        """
        num_species = self.num_species
        num_reaction = self.num_reaction
        num_node = num_species + num_reaction
        graph_dct:dict = {i: [] for i in range(num_node)}
        for i_reaction in range(num_reaction):
            for i_species in range(num_species):
                # Add an edge for each reactant
                species_vtx = i_species
                reaction_vtx = num_species + i_reaction
                if identity == cn.ID_STRONG:
                    # Reactants
                    for _ in range(int(self.reactant_mat.values[i_species, i_reaction])):
                        graph_dct[species_vtx].append(reaction_vtx)
                    # Products
                    for _ in range(int(self.product_mat.values[i_species, i_reaction])):
                        graph_dct[reaction_vtx].append(species_vtx)
                else:
                    # Weak identity. Use the standard stoichiometry matrix
                    stoichiometry = int(self.standard_mat.values[i_species, i_reaction])
                    for _ in range(np.abs(stoichiometry)):
                        if stoichiometry > 0:
                            # Product
                            graph_dct[reaction_vtx].append(species_vtx)
                        elif stoichiometry < 0:
                            # Reactant
                            graph_dct[species_vtx].append(reaction_vtx)
                        else:
                            pass
        return graph_dct
    
    def makePynautyNetwork(self, identity=cn.ID_STRONG)->Graph:
        """
        Make a pynauty graph from the network. Species are indices 0 to num_species - 1 and reactions are
        indices num_species to num_species + num_reaction - 1.

        Args:
            identity (str): cn.ID_STRONG or cn.ID_WEAK

        Returns:
            Graph: Pynauty graph
        """
        graph_dct = self.getGraphDct(identity=identity)
        graph = Graph(len(graph_dct.keys()), directed=True)
        for node, neighbors in graph_dct.items():
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
        graph_dct = self.getGraphDct(identity=identity)
        outputs = []
        #
        for source, destinations in graph_dct.items():
            for destination in destinations:
                outputs.append(f"{source}>{destination}")
        return '\n'.join(outputs)