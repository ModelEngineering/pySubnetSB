# DATA SETS
# Provenance
Data sets are produced by the functions in ``make_data.py`` in the ``scripts`` directory. A couple of the functions are compute intensive, and so have been parallelized. Some manual work is needed to set up parallel runs. Also, to improve the quality of the estimates of probability of occurrence, there may be multiple runs done for ``biomodels_summary`` that should be combined (to calculate mean values) using ``consolidateBiomodelSummary`` in ``merge_data``. Probability of occurrence calculations are only done for smaller models, those with no more than 10 reactions. (See ``makeModelSummary`` to update this threshold.)
# Data types
* ModelName - String that identifies a model. In BioModels, this is the BioModels model name.
* Network - String containing Antiimony representation of the network (without rate laws and constant initializations)
* Count - Non-negative integer
* Probability - Floating point number between 0 and 1.
* AssignmentPair - JSON structure describing the assignment of target species and reactions to reference species and reactions
* Bool - binary

# Data sets and their columns
* ``full_biomodels_strong.csv``: Results of using models with no more than 10 reactions as reference and larger models as targets using strong identity
  * ``reference_name`` (ModelName)
  * ``target_name`` (ModelName)
  * ``reference_network`` (Network)
  * ``induced_network``(Network) Subnet of target that corresponds to the reference network
  * ``assignment_pairs`` (list-AssignmentPair)
  * ``significance_level`` (Probability) Approximate probability of the induced network for random target and reference networks of the same sizes
  * ``is_truncated`` (Bool) True if the number of assignments satisfying constraints exceeds 10**11 in determining if there is an induced network
* ``subnet_biomodels_strong.csv`` Subset of ``full_biomodels_strong.csv`` that have induced networks. Also have:
  * ``truncated_strong`` truncated in the calculation of probability of occurrence with identity=strong
  * ``truncated_weak`` truncated in the calculation of probability of occurrence with identity=weak
  * ``probability_of_occurrence_strong`` probability of occurrence using strong identity
  * ``probability_of_occurrence_weak`` probability of occurrence using weak identity
* ``full_biomodels_weak.csv`` (same as ``biomodels_subnet_strong.csv`` but for weak identity)
* ``subnet_biomodels_weak.csv`` Subset of ``full_biomodels_weak.csv`` that have induced networks
* ``biomodels_summary.csv``. Same columns as ``subnet_biomodels_strong.csv``
  * ``model_name`` (ModelName)
  * ``num_species`` (Count)
  * ``num_reaction`` (Count)
  * ``probability_of_occurrence`` (Probability)
* ``biomodels_serialized.txt``: serialization file for BioModels in format used in APIs. Each line is a serialization for a model. Lines can be added and/or deleted.
* ``oscillators_serialized.txt``: serialization file for oscillators in same format as ``biomodels_serialized.txt``.
