#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    **************************************************
    Parset to COPPER pre-processing Configuration file
    **************************************************
"""


__author__ = 'Alan Loh'
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    "_ParsetProperty",
    "CopperConfig",
    "Parset"
]


from collections.abc import MutableMapping
from astropy.time import Time
import astropy.units as u
import numpy as np
import logging
import sys
import os
import re
import glob
from typing import List

from . import (
    ENV_FILE_PATH,
    AVAILABLE_STAT,
    AVERAGE_TIMESTEP_MIN,
    AVERAGE_TIMESTEP_MAX,
    AVERAGE_FREQSTEP_MIN,
    DEFAULT_STARTCHAN,
    FLAG_STRATEGY_FILE_PATH
)


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


# ============================================================= #
# ---------------------- _ParsetProperty ---------------------- #
# ============================================================= #
class _ParsetProperty(MutableMapping):
    """ Class which mimics a dictionnary object, adapted to
        store parset metadata per category. It understands the
        different data types from raw strings it can encounter.
    """

    def __init__(self, data=()):
        self.mapping = {}
        self.update(data)

    def __getitem__(self, key):
        return self.mapping[key]

    def __delitem__(self, key):
        del self.mapping[key]

    def __setitem__(self, key, value):
        """
        """
        value = value.replace('\n', '')
        value = value.replace('"', '')

        if value.startswith('[') and value.endswith(']'):
            # This is a list
            val = value[1:-1].split(',')
            value = []
            # Parse according to syntax
            for i in range(len(val)):
                if '..' in val[i]:
                    # This is a subband syntax
                    subBandSart, subBanStop = val[i].split('..')
                    value.extend(
                        list(
                            range(
                                int(subBandSart),
                                int(subBanStop) + 1
                            )
                        )
                    )
                elif ':' in val[i]:
                    # Might be a time object
                    try:
                        item = Time(val[i].strip(), precision=0)
                    except ValueError:
                        item = val[i]
                    value.append(item)
                elif val[i].isdigit():
                    # Integers (there are not list of floats)
                    value.append(int(val[i]))
                else:
                    # A simple string
                    value.append(val[i])

        elif value.lower() in ['on', 'enable', 'true']:
            # This is a 'True' boolean
            value = True

        elif value.lower() in ['off', 'disable', 'false']:
            # This is a 'False' boolean
            value = False
        
        elif 'angle' in key.lower():
            # This is a float angle in degrees
            value = float(value) * u.deg
        
        elif value.isdigit():
            value = int(value)
        
        elif ':' in value:
            # Might be a time object
            try:
                value = Time(value.strip(), precision=0)
            except ValueError:
                pass

        else:
            pass
        
        self.mapping[key] = value

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f'{type(self).__name__}({self.mapping})'
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ----------------------- CooperConfig ------------------------ #
# ============================================================= #
class CopperConfig:
    """ """

    def __init__(self, file_name: str, channelization: int, dumptime: int, subbands: np.ndarray, email: str):
        self.file_name = file_name
        self.channelization = channelization
        self.dumptime = dumptime
        self.subbands = subbands
        self.email = email
        self._quality_step = True

        self.data = {
            "worker": {
                "env_file": {
                    "value": None,
                    "default": "env_default.sh",
                    "check_function": self._check_env_file
                }
            },
            "process": {
                "avg_timestep": {
                    "value": None,
                    "default": 8,
                    "parsing_function": (lambda x: int(x)),
                    "check_function": self._check_avg_timestep,
                },
                "avg_freqstep": {
                    "value": None,
                    "default": 6,
                    "parsing_function": (lambda x: int(x)),
                    "check_function": self._check_avg_freqstep,
                },
                "startchan": {
                    "value": None,
                    "default": DEFAULT_STARTCHAN,
                    "parsing_function": (lambda x: int(x)),
                    "check_function": self._check_startchan,
                },
                "nchan": {
                    "value": None,
                    "default": self.channelization - DEFAULT_STARTCHAN*2,
                    "parsing_function": (lambda x: int(x)),
                    "check_function": self._check_nchan
                },              
                "compress": {
                    "value": None,
                    "default": False,
                    "parsing_function": (lambda x: False if x.lower()=="false" else True),
                    "check_function": self._check_compress,
                },
                "flag_strategy": {
                    "value": None,
                    "default": 'NenuFAR-64C1S.rfis',# --> repo: "...config" --> chercherer *.rfis/.lua
                    "check_function": self._check_flag_strategy,
                }
            },
            "quality": {
                "sws": {
                    "value": None,
                    "default": None,
                    "parsing_function": (lambda x: str([f"SW{i+1:02}-{val[0]}-{val[1]}" for i, val in enumerate(re.findall(r"(\d+)-(\d+)", x))])),
                    "check_function": self._check_sws,
                },
                "stat_pols": {
                    "value": None,	
                    "default": None,
                    "check_function": self._check_stat_pols,
                }
            }
        }


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def file_name(self) -> str:
        """ """
        return self._file_name
    @file_name.setter
    def file_name(self, f: str) -> None:
        log.info(f"Initializing the writing of {f}...")
        directory = os.path.abspath(os.path.dirname(f))
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"The directory '{directory}' does not exist. Impossible to write '{f}'."
            )
        self._file_name = f
    

    @property
    def email(self) -> List[str]:
        """ """
        return self._email.split(",")
    @email.setter
    def email(self, e: str) -> None:
        self._email = e


    @property
    def tasks(self) -> str:
        """ """
        quality = self.data["quality"]
        if (quality["sws"]["value"] is None) or (quality["stat_pols"]["value"] is None):
            self._quality_step = False
            log.info("No quality step will be applied.")
            return "tasks = ['process', 'rsync']"
        else:
            return "tasks = ['process', 'rsync_quality', 'quality', 'rsync']"


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #

    def default_configuration(self):
        """ """
        log.info("Applying default pre-processing...")
        self._write_file(kind="default")
    

    def custom_configuration(self, phase_center: _ParsetProperty):
        """ """
        # Decode the parameter field
        pattern = r"(\S+)\s*=\s*(.*?)\s*(?=\S+\s*=|$)"
        parameters = dict(re.findall(pattern, phase_center["parameters"]))

        self._set_parameters(parameters)
        self._check_parameters()
        self._write_file(kind="value")


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _write_file(self, kind: str = "value") -> None:
        """ """
        with open(self.file_name, "w") as wfile:
            wfile.write(self.tasks + "\n\n")
           
            wfile.write(f"log_email = {self.email}\n")
           
            wfile.write("\n[worker]\n")
            for key in self.data["worker"]:
                wfile.write(f"{key} = {self.data['worker'][key][kind]}\n")
           
            wfile.write("\n[process]\n")
            for key in self.data["process"]:
                wfile.write(f"{key} = {self.data['process'][key][kind]}\n")
           
            if self._quality_step:
                wfile.write("\n[quality]\n")
                for key in self.data["quality"]:
                    wfile.write(f"{key} = {self.data['quality'][key][kind]}\n")

        log.info(f"'{self.file_name}' written.")


    def _set_parameters(self, parameters: dict) -> None:
        """ """
        log.info("Setting the parameters found in the parset file...")
        for key in parameters.keys():
            key_lower = key.lower()
            for step in self.data.keys():
                step_dict = self.data[step]
                if key_lower in step_dict.keys():
                    value = parameters[key]
                    try:
                        # If the value needs to be parsed
                        parsing = step_dict[key_lower]["parsing_function"]
                        value = parsing(value)
                    except KeyError:
                        pass
                    except:
                        log.warning(f"Parameter '{key}': parsing error. Considering no value.")
                        value = None
                    step_dict[key_lower]["value"] = value
                    log.info(f"'{key_lower}' set to '{value}'.")
                    break
            else:
                log.warning(
                    f"Unexpected parset parameter key '{key}': skipped."
                )


    def _check_parameters(self) -> None:
        """ """
        log.info("Checking the parameters values...")
        for step in self.data.keys():
            step_dict = self.data[step]
            for key in step_dict.keys():
                # Set the parameters to its default value if it has not been filled
                if step_dict[key]["value"] is None:
                    step_dict[key]["value"] = step_dict[key]["default"]
                    log.info(f"Parameter '{key}': missing. Set to default value {step_dict[key]['default']}.")

                # Check that the parameters is filled as expected
                verify = step_dict[key]["check_function"]
                if not verify(step_dict[key]["value"]):
                    log.warning(
                        f"Parameter '{key}': invalid value. Set to default value {step_dict[key]['default']}."
                    )
                    step_dict[key]["value"] = step_dict[key]["default"]


    @staticmethod
    def _check_env_file(file_name: str) -> bool:
        """ """
        available_env_files = glob.glob(os.path.join(ENV_FILE_PATH, "*.sh"))
        file_exists = file_name in available_env_files
        if not file_exists:
            log.warning(
                f"Unable to find 'env_file': '{file_name}' among the existing '{available_env_files}'."
            )
        return file_exists


    @staticmethod
    def _check_avg_timestep(timestep: int) -> bool:
        """ """
        return (timestep >= AVERAGE_TIMESTEP_MIN) & (timestep <= AVERAGE_TIMESTEP_MAX)


    def _check_avg_freqstep(self, freqstep: int) -> bool:
        """ """
        return (freqstep >= AVERAGE_FREQSTEP_MIN) & (freqstep <= self.channelization)


    def _check_startchan(self, startchan: int) -> bool:
        """ """
        return (self.channelization - startchan*2) >= 1


    def _check_nchan(self, nchan: int) -> bool:
        """ """
        return (nchan > 0) & (nchan < self.channelization - 1)


    @staticmethod
    def _check_compress(compress: bool) -> bool:
        """ """
        return isinstance(compress, bool)


    @staticmethod
    def _check_flag_strategy(flag_strategy: str) -> bool:
        """ """
        available_strategies = glob.glob(os.path.join(FLAG_STRATEGY_FILE_PATH, "*.rfis"))
        available_strategies += glob.glob(os.path.join(FLAG_STRATEGY_FILE_PATH, "*.lua"))
        file_exists = flag_strategy in available_strategies
        if not file_exists:
            log.warning(
                f"Unable to find 'flag_strategy': '{flag_strategy}' among the existing '{available_strategies}'."
            )
        return file_exists


    def _check_sws(self, sws: str) -> bool:
        """ """
        matches = re.findall(r"(\d+)-(\d+)-(\d+)", sws)
        for edges in matches:
            print(edges)
            low_edge = int(edges[1])
            high_edge = int(edges[2])
            if not np.any((self.subbands >= low_edge) * (self.subbands <= high_edge)):
                log.warning(
                    f"No subbands in the desired quality interval {low_edge}-{high_edge}."
                )
                return False
        else:
            return True


    @staticmethod
    def _check_stat_pols(stat_pol: str) -> bool:
        """ """
        stat_pol = stat_pol.replace("[", "").replace("]", "").replace(" ", "").split(",")
        known_strategies = np.all(np.isin(stat_pol, AVAILABLE_STAT))
        if not known_strategies:
            log.warning(
                f"Some quality strategies are not among the known '{AVAILABLE_STAT}'."
            )
        return known_strategies
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------------- Parset -------------------------- #
# ============================================================= #
class Parset(object):
    """
    """

    def __init__(self, parset):
        self.observation = _ParsetProperty()
        self.output = _ParsetProperty()
        self.anabeams = {} # dict of _ParsetProperty
        self.digibeams = {} # dict of _ParsetProperty
        self.phase_centers = {}
        self.parset_user = ""
        self.parset = parset


    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def parset(self):
        """
        """
        return self._parset
    @parset.setter
    def parset(self, p):
        if not isinstance(p, str):
            raise TypeError(
                'parset must be a string.'
            )
        if not p.endswith('.parset'):
            raise ValueError(
                'parset file must end with .parset'
            )
        p = os.path.abspath(p)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f'Unable to find {p}'
            )
        self._parset = p
        self._decodeParset()


    @property
    def version(self) -> tuple:
        """ """
        version_str = self.observation.get("parsetVersion", "0")
        version_tuple = tuple(map(lambda x: int(x), version_str.split(".")))
        return version_tuple


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def to_config_tml(self, directory: str = ""):
        """ Converts the Parset into a COPPER pre-processing configuration file.
        """
        file_name = os.path.basename(self.parset).replace(".parset", ".tml")
        config = CopperConfig(
            file_name=os.path.join(directory, file_name),
            channelization=self.output["nri_channelization"],
            dumptime=self.output["nri_dumpTime"],
            subbands=np.array(self.phase_centers[0]["subbandList"]),
            email=self.observation["contactEmail"]
        )

        # Fill the CopperConfig object with relevant values from the parset to check the configuration
        # config.blabla()

        # If the parset version is older than 1.0, the PhaseCenters
        # were not defined, it was impossible to specify parameters.
        if self.version < (1, 0):
            log.warning(f"Parset version '{self.version}' does not contain PhaseCenters.")
            return config.default_configuration()

        phase_center = self.phase_centers[0]
        # If there is no PhaseCenter[0].parameters
        if "parameters" not in phase_center:
            log.warning("No 'parameters' found for PhaseCenter[0], or no PhaseCenter found at all.")
            return config.default_configuration()
        
        return config.custom_configuration(phase_center)


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _decodeParset(self) -> None:
        """
        """
        
        with open(self.parset, 'r') as file_object:
            line = file_object.readline()
            
            while line:
                try:
                    dicoName, content = line.split('.', 1)
                except ValueError:
                    # This is a blank line
                    pass
                
                key, value = content.split('=', 1)
                
                if line.startswith('Observation'):
                    self.observation[key] = value
                
                elif line.startswith('Output'):
                    self.output[key] = value
                
                elif line.startswith('AnaBeam'):
                    anaIdx = int(re.search(r'\[(\d*)\]', dicoName).group(1))
                    if anaIdx not in self.anabeams.keys():
                        self.anabeams[anaIdx] = _ParsetProperty()
                        self.anabeams[anaIdx]['anaIdx'] = str(anaIdx)
                    self.anabeams[anaIdx][key] = value
                
                elif line.startswith('Beam'):
                    digiIdx = int(re.search(r'\[(\d*)\]', dicoName).group(1))
                    if digiIdx not in self.digibeams.keys():
                        self.digibeams[digiIdx] = _ParsetProperty()
                        self.digibeams[digiIdx]['digiIdx'] = str(digiIdx)
                    self.digibeams[digiIdx][key] = value
                
                elif line.startswith('PhaseCenter'):
                    pcIdx = int(re.search(r'\[(\d*)\]', dicoName).group(1))
                    if pcIdx not in self.phase_centers.keys():
                        self.phase_centers[pcIdx] = _ParsetProperty()
                        self.phase_centers[pcIdx]['pcIdx'] = str(pcIdx)
                    self.phase_centers[pcIdx][key] = value

                line = file_object.readline()
            
            log.info(
                f"Parset '{self._parset}' loaded."
            )
        
        try:
            with open(self.parset + "_user", "r") as file_object:
                line = file_object.readline()
                while line:
                    self.parset_user = self.parset_user + line
                    line = file_object.readline()
        except Exception as e:
            pass
# ============================================================= #
# ============================================================= #

