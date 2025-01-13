'''A ConstraintOption is a set of boolean options for constraints.'''

import pySubnetSB.constants as cn # type: ignore
import pySubnetSB.util as util  # type: ignore


import string  # type: ignore
from typing import List


#####################################
class ConstraintOptionCollection(object): 

    # Subclasses for species and reactions

    def __init__(self, **kwargs):
        """
        kwargs:
            names of boolean options and their initial values
        """
        self.__dict__.update(kwargs)
        self.option_names = list(kwargs.keys())
        self.long_to_short_dct = {n: self.short_names[i] for i, n in enumerate(self.option_names)}
        self.short_to_long_dct = {v: k for k, v in self.long_to_short_dct.items()}
        self.num_option = len(self.option_names)

    def copy(self)->'ConstraintOptionCollection':
        """
        Returns a copy of the current object.
        """
        dct = {k: self.__dict__[k] for k in self.option_names}
        new_option = self.__class__(**dct)
        return new_option
    
    def __eq__(self, other)->bool:
        # Returns True if all constraint options are the same.
        for option_name in self.option_names:
            if self.__dict__[option_name] != other.__dict__[option_name]:
                return False
        return True

    def __repr__(self)->str:
        return str(self.__class__) + ": " + ', '.join(self.getTrueNames())
    
    def getTrueNames(self)->list[str]:
        results:list = []
        for i, key in enumerate(self.option_names):
            if self.__dict__[key]:
                results.append(key)
        return results

    @property 
    def collection_short_name(self)->str:
        # Creates a short name for the collection based on the options that are True.
        names = self.getTrueNames()
        if len(names) == 0:
            return cn.NONE
        return '+'.join([self.long_to_short_dct[n] for n in names])
    
    @property 
    def long_name(self)->str:
        # Returns a combination of short names to describe the current state
        names = self.getTrueNames()
        if len(names) == 0:
            return cn.NONE
        return '+'.join(names)
    
    def setAllFalse(self)->None:
        """
        Set all values to False.
        """
        for option_name in self.option_names:
            self.__dict__[option_name] = False

    def makeOptionFromShortNames(self, short_names:List[str])->'ConstraintOptionCollection':
        """
        Returns a new ConstraintOption object with the short names set to True.

        Args:
            short_names: List[str] Short names of the options to set to True.
        """
        long_names = [self.short_to_long_dct[n] for n in short_names]
        new_option = self.copy()
        new_option.setAllFalse()
        for name in long_names:
            new_option.__dict__[name] = True
        return new_option
    
    def makeFromCollectionShortName(self, collection_short_name:str)->'ConstraintOptionCollection':
        """
        Returns a new ConstraintOptionCollection with the options specified by the collection short name.

        Args:
            collection_short_name (str): _description_

        Returns:
            ConstraintOptionCollection: _description_
        """
        option_collection = self.copy()
        short_names = collection_short_name.split('+')
        option_collection.setAllFalse()
        for short_name in short_names:
            option_collection.__dict__[self.short_to_long_dct[short_name]] = True
        return option_collection
    
    def isTrue(self, option_name:str)->bool:
        """
        Returns True if the option is True.

        Args:
            option_name (str): _description_

        Returns:
            bool: _description_
        """
        if not option_name in self.__dict__.keys():
            return False
        return self.__dict__[option_name]

    def iterator(self):
        """
        Iteratively returns an OptionCollection of the same type with all combinations of
        values for the boolean attributes.
        """
        name_collections = util.getAllSubsets(self.short_names)
        for name_collection in name_collections:
            yield self.makeOptionFromShortNames(name_collection)


#####################################
class ReactionConstraintOptionCollection(ConstraintOptionCollection): 

    def __init__(self,
                 #is_make_successor_predecessor_constraint_matrix:bool=True,
                 is_make_n_step_constraint_matrix:bool=True,
                 is_make_classification_constraint_matrix:bool=True,
                 #is_make_autocatalysis_constraint_matrix:bool=True,
                 ):
        self.short_names = list(string.ascii_letters[:26])
        self.short_names.reverse()
        self.short_names = self.short_names[:2]
        self.short_names.reverse()
        super().__init__(
            #is_make_successor_predecessor_constraint_matrix=is_make_successor_predecessor_constraint_matrix,
            is_make_n_step_constraint_matrix=is_make_n_step_constraint_matrix,
            is_make_classification_constraint_matrix=is_make_classification_constraint_matrix,
            #is_make_autocatalysis_constraint_matrix=is_make_autocatalysis_constraint_matrix,
            )
        

#####################################
class SpeciesConstraintOptionCollection(ConstraintOptionCollection): 


    def __init__(self, 
                 #is_reactant_product_count_constraint_matrix:bool=True,
                 #is_autocatalysis_constraint:bool=True,
                 is_reactant_product_constraint_matrix:bool=True,
                 #is_successor_predecessor_constraint_matrix:bool=True,
                 is_n_step_constraint_matrix:bool=True):
        self.short_names = list(string.ascii_letters[:2])
        super().__init__(
              #is_reactant_product_count_constraint_matrix=is_reactant_product_count_constraint_matrix,
              #is_autocatalysis_constraint=is_autocatalysis_constraint,
              #is_successor_predecessor_constraint_matrix=is_successor_predecessor_constraint_matrix,
              is_n_step_constraint_matrix=is_n_step_constraint_matrix,
              is_reactant_product_constraint_matrix=is_reactant_product_constraint_matrix,
        )