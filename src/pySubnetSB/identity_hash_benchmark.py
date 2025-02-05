'''Evaluates collisions in the identity hash.'''

import pySubnetSB.constants as cn # type: ignore
from pySubnetSB.network import Network      # type: ignore

import collections
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from typing import List, Optional

NUM_NETWORK = 'num_network'
NUM_COLLISION = 'num_collision'
NUM_UNIQUE_HASH = 'num_unique_hash'
MAX_NETWORK_IN_HASH = 'max_network_in_hash'
NUM_SPECIES = 'num_species'
NUM_REACTION = 'num_reaction'
PERCENT_COLLISION = 'percent_collision'
#  num_network: number of networks simulated
#  num_collision: number of collisions in total
#  num_unique_hash: number of distinct hash values
#  max_network_in_hash: maximum number of networks in a hash
#  num_species: number of species
#  num_reaction: number of reactions
#  frac_collision: fraction of collisions
COLUMNS = [NUM_SPECIES, NUM_REACTION, NUM_NETWORK, NUM_COLLISION, NUM_UNIQUE_HASH, MAX_NETWORK_IN_HASH,
      PERCENT_COLLISION]
TITLE_FONT_SIZE_INCREMENT = 2
NAME_DCT = {NUM_SPECIES: 'number of species', NUM_REACTION: 'number of reactions',
            NUM_COLLISION: 'number collision',
      NUM_UNIQUE_HASH: 'number unique hash', MAX_NETWORK_IN_HASH: 'max network in hash',
      PERCENT_COLLISION: 'percent collision'}


class IdentityHashBenchmark(object):
    def __init__(self, species_nums:List[int], reaction_nums:List[int], num_network:int=1000):
        """
        Args:
            species_nums: number of species
            reaction_nums: number of reactions
            num_iteration: number of networks simulated for a combination of species and reactions
        """
        self.species_nums = species_nums
        self.reaction_nums = reaction_nums
        self.num_network = num_network

    def calculateHashStatistics(self)->pd.DataFrame:
        """
        Calculates the number of collisions in the identity hash.

        Randomly generates networks and calculates the number of collisions in the identity hash.
        Eliminates strongly identical networks from collisions.

        Returns:
            pd.DataFrame: dataframe
                num_species: number of species
                num_reaction: number of reactions
                num_network: number of networks simulated
                num_collision: number of collisions in total
                num_unique_hash: number of distinct hash values
                max_network_in_hash: maximum number of networks in a hash
        """
        # Initialize result dictionaries
        dct:dict = {c: [] for c in COLUMNS}
        for num_species in self.species_nums:
            for num_reaction in self.reaction_nums:
                networks = [Network.makeRandomNetworkByReactionType(
                    num_species=num_species, num_reaction=num_reaction)
                      for _ in range(self.num_network)]
                # Group networks by hash
                hash_dct:dict = {}
                for network in networks:
                    if network.network_hash in hash_dct:
                        hash_dct[network.network_hash].append(network)
                    else:
                        hash_dct[network.network_hash] = [network]
                # Eliminate duplicates
                unduplicated_network_dct:dict = {}
                for key in hash_dct.keys():
                    unduplicated_network_dct[key] = [hash_dct[key][0]]
                    for same_hash_network in hash_dct[key][1:]:
                        result = same_hash_network.isStructurallyIdentical(unduplicated_network_dct[key][0],
                                is_subnet=False, identity=cn.ID_STRONG)
                        if result.is_truncated:
                            print('Truncated')
                        if not result:
                            unduplicated_network_dct[key].append(same_hash_network)
                # Calculate statistics
                count_arr = np.array([len(v) for v in unduplicated_network_dct.values()])
                max_network_in_hash = max(count_arr)
                num_collision = np.sum([c for c in count_arr if c > 1])
                num_unique_hash = len(count_arr)
                # Record the results
                dct[NUM_SPECIES].append(num_species)
                dct[NUM_REACTION].append(num_reaction)
                dct[NUM_NETWORK].append(self.num_network)
                dct[NUM_COLLISION].append(num_collision)
                dct[NUM_UNIQUE_HASH].append(num_unique_hash)
                dct[MAX_NETWORK_IN_HASH].append(max_network_in_hash)
                percent_collision = round(100*num_collision / np.sum(count_arr))
                dct[PERCENT_COLLISION].append(percent_collision)
        # Create the dataframe
        df = pd.DataFrame(dct)
        return df
    
    def plotHashStatistics(self, statistic_name:str=PERCENT_COLLISION, title:str="",
          font_size:int=12, hash_statistics_df:Optional[pd.DataFrame]=None, is_plot=True)->pd.DataFrame:
        """
        Plots the number of collisions in the identity hash.

        Args:
            statistic_name: column name in the dataframe
            title: title of the plot
            font_size: font size
            hash_statistics_df: dataframe for data used
            is_plot: if True, plot the DatFrame
        """
        if hash_statistics_df is None:
            hash_statistics_df = self.calculateHashStatistics()
        # Create a pivot table
        plot_df = hash_statistics_df.pivot_table(index=NUM_SPECIES,
              columns=NUM_REACTION, values=statistic_name)
        plot_df = plot_df.sort_index(ascending=False)
        # Plot the heatmap
        ax = sns.heatmap(plot_df, cmap='Reds', vmin=0, vmax=100,
                          annot_kws={'size': font_size}, annot=True,
                          cbar_kws={'label': NAME_DCT[statistic_name]})
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_xlabel(NAME_DCT[NUM_REACTION], size=font_size)
        ax.set_ylabel(NAME_DCT[NUM_SPECIES], size=font_size)
        ax.set_title(title, size=font_size + TITLE_FONT_SIZE_INCREMENT)
        ax.figure.axes[-1].yaxis.label.set_size(font_size)
        cbar_ticklabels = ax.figure.axes[-1].get_yticklabels()
        ax.figure.axes[-1].set_yticklabels(cbar_ticklabels, size=font_size)
        if is_plot:
            plt.show()
        return hash_statistics_df