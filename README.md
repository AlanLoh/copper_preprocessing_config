# copper_preprocessing_config

Create the COPPER pre-processing configuration files from a NenuFAR parset



```python

from copperconfig.nenufar_parset import Parset

parset = Parset("<parset_file>")
parset.to_config_tml(directory="")

```