# DATA SETS

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
  * ``is_truncated`` (Bool) True if the number of assignments satisfying constraints exceeds 10**11.
* ``subnet_biomodels_strong.csv`` Subset of ``full_biomodels_strong.csv`` that have induced networks
* ``full_biomodels_weak.csv`` (same as ``biomodels_subnet_strong.csv`` but for weak identity)
* ``subnet_biomodels_weak.csv`` Subset of ``full_biomodels_weak.csv`` that have induced networks
* ``biomodels_summary.csv``
  * ``model_name`` (ModelName)
  * ``num_species`` (Count)
  * ``num_reaction`` (Count)
  * ``probability_of_occurrence`` (Probability)
* ``biomodels_serialized.txt``: serialization file for BioModels in format used in APIs. Each line is a serialization for a model. Lines can be added and/or deleted.
* ``oscillators_serialized.txt``: serialization file for oscillators in same format as ``biomodels_serialized.txt``.
