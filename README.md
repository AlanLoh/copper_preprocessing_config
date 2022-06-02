# copper_preprocessing_config

Create the COPPER pre-processing configuration files from a NenuFAR parset


## Set the environment

Global parameters can be defined in `copperconfig/__init__.py`, which contains by default:

```python
# Environnement file (*.sh) path 
ENV_FILE_PATH = "./"

# Flag strategy (*.rfis / *.lua) path
FLAG_STRATEGY_FILE_PATH = "./"
DEFAULT_FLAG_RFI = True
DEFAULT_FLAG_MEMORYPERC = 30 # not set via parameters

# Parameters Checks
AVAILABLE_STAT = ["SNR_XX", "SNR_YY", "RFIPercentage_XX"]

DEFAULT_ENV_FILE = "env_default.sh"

AVERAGE_TIMESTEP_MIN = 1
AVERAGE_TIMESTEP_MAX = 60
DEFAULT_AVERAGE_TIMESTEP = 8

AVERAGE_FREQSTEP_MIN = 1
DEFAULT_AVERAGE_FREQSTEP = 6

DEFAULT_STARTCHAN = 0
```

## Run the code

### Python interpreter

```python
from copperconfig.nenufar_parset import Parset

parset = Parset("<parset_file>")
parset.to_config_toml(directory="")
```

### Command line

```
parset2preprocconfig --parset <nenufar_parset.parset> --directory <path_to_store_the_toml_file>
```

## NenuFAR Parset file

The expected syntax for the `parameters` field of a `PhaseCenter` block is:

```
...
PhaseCenter[0].parameters="env_file=env_ES04.sh avg_timestep=1 avg_freqstep=15 startchan=2 nchan=60 compress=false flag_strategy=NenuFAR-64C1S.rfis sws=[106-200,202-300,306-418] stat_pols=[SNR_XX,SNR_YY,RFIPercentage_XX]"
...
```
