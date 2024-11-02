"""Analyzes biomodels for subnets."""

from sirn.subnet_finder import SubnetFinder  # type: ignore

from multiprocessing import freeze_support

def main():
    df = SubnetFinder.findBiomodelsSubnet(reference_model_size=10,
              batch_size=1, is_initialize=False, is_report=True)
    df.to_csv("biomodels_subnet.csv", index=False)

if __name__ == "__main__":
    freeze_support()
    main()