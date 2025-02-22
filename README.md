[![Build](https://github.com/ModelEngineering/pySubnetSB/actions/workflows/github-actions.yml/badge.svg)](https://github.com/ModelEngineering/pySubnetSB/actions/workflows/github-actions.yml)
[![Coverage](https://github.com/ModelEngineering/pySubnetSB/actions/workflows/github-actions.yml/coverage.svg)](https://github.com/ModelEngineering/pySubnetSB/actions/workflows/github-actions.yml)

# SUBNET DISCOVERY FOR SBML MODELS

# Motivation
Many advances in biomedical research are driven by structural analysis, a study of the interconnections
between elements in biological systems (e.g., identifying drug target and phylogenetic analyses). Structural analysis
appeals because structural information is much easier to obtain than dynamical data such as species concentrations
and reaction fluxes. Our focus is on subnet discovery in chemical reaction networks (CRNs); that is, discovering a
subset of a target CRN that is structurally identical to a reference CRN. Applications of subnet discovery include the
discovery of conserved chemical pathways and the elucidation of the structure of complex CRNs. Although there are
theoretical results for finding subgraphs, we are unaware of tools for CRN subnet discovery. This is in part due to the
special characteristics of CRN graphs, that they are directed, bipartite, hypergraphs.

# Results
We introduces pySubnetSB, an open source python package for discovering subnets represented in the systems
biology markup language (SBML) community standard. pySubnetSB uses a constraint-based approach to discover
subgraphs using techniques that work well for CRNs, and provides considerable speed-up through vectorization and
process-based parallelism. We provide a methodology for evaluating the statistical significance of subnet discovery and
apply pySubnetSB to discovering subnets in more than 100,000 model pairs in the BioModels repository of curated
models.

# Availability
pySubnetSB is installed using

    pip install pySubnetSB
https://github.com/ModelEngineering/pySubnetSB/blob/main/
examples/api_basics.ipynb is a Jupyter notebook that demonstrates pySubsetSB capabilities.

# Version History
* 0.1.0 2/21/2025. First beta release.