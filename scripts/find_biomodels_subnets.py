"""Analyzes biomodels for subnets."""

from sirn.subnet_finder import SubnetFinder  # type: ignore

df = SubnetFinder.findBiomodelsSubnet(reference_model_size=10,
              batch_size=2, is_initialize=False, is_report=True)
df.to_csv("biomodels_subnet.csv", index=False)

