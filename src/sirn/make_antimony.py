'''Creates an Antimony model from a reference.'''

import tellurium as te  # type: ignore

ANT = "ant"
XML = "xml"

def makeAntimony(model_reference:str)->str:
    """
    Creates a roadrunner instance from a model reference.

    Parameters
    ----------
    model_reference: 
        URL (http:...)
        Antimony file (extension is .ant)
        XML file (extension is .xml)
        XML string
        Antimony string
    
    Returns
    -------
    ExtendedRoadrunner object
    """
    if not isinstance(model_reference, str):
        raise ValueError("Invalid model reference")
    #
    if model_reference[0:4] == "http":
        # URL
        roadrunner = te.loadSBMLModel(model_reference)
    else:
        parts = model_reference.split(".")
        if len(parts) == 2:
            # XML file
            if parts[1] == XML:
                roadrunner = te.loadSBMLModel(model_reference)
            elif parts[1] == ANT:
                # Antimony file
                roadrunner = te.loadAntimonyModel(model_reference)
            else:
                # Assume string for antimony model
                if len(model_reference) > 10:
                    if XML in model_reference[0:10]:
                        roadrunner = te.loads(model_reference)
                else:
                    # Possible Antimony string
                    return model_reference
        else:
            if XML in model_reference[0:10]:
                roadrunner = te.loads(model_reference)
            else:
                roadrunner = te.loada(model_reference)
    return roadrunner.getAntimony()
