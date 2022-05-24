#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import pytest
from unittest.mock import patch, mock_open, call
import os
from copperconfig.nenufar_parset import Parset


DATA_DIR = os.path.join(os.path.dirname(__file__), "parsets")


# ============================================================= #
# ---------------- test_fixedtarget_properties ---------------- #
# ============================================================= #
def test_parset_reading():
    parset_file = os.path.join(DATA_DIR, "sun.parset")
    parset = Parset(parset_file)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# ---------------- test_fixedtarget_properties ---------------- #
# ============================================================= #
def test_parset_converting():
    parset_file = os.path.join(os.path.dirname(__file__), "parsets/sun.parset")
    parset = Parset(parset_file)

    assert parset.phase_centers[0]["toDo"] == "NICKEL"

    parset.to_config_toml()
# ============================================================= #
# ============================================================= #

# ============================================================= #
# --------------------- test_tml_writing ---------------------- #
# ============================================================= #
def test_tml_writing():
    parset_file = os.path.join(DATA_DIR, "sun.parset")
    parset = Parset(parset_file)

    open_mock = mock_open()
    with patch("copperconfig.nenufar_parset.open", open_mock, create=True):
        parset.to_config_toml()

    open_mock.assert_called_with("sun.toml", "w")
    calls = [call(
        "tasks = ['process', 'rsync_quality', 'quality', 'rsync']\n\n"
        "log_email = alan.loh@obspm.fr\n"
        '\n[worker]\n'
        'env_file = env_default.sh\n'
        '\n[process]\n'
        'avg_timestep = 1\n'
        'avg_freqstep = 15\n'
        'startchan = 2\n'
        'nchan = 60\n'
        'compress = False\n'
        'flag_strategy = NenuFAR-64C1S.rfis\n'
        '\n[quality]\n'
        "sws = ['SW01-106-200', 'SW02-202-300', 'SW03-306-418']\n"
        "stat_pols = ['SNR_XX', 'SNR_YY', 'RFIPercentage_XX']\n"
    )]
    open_mock.return_value.write.assert_has_calls(calls)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- test_empty_param ---------------------- #
# ============================================================= #
def test_empty_param():
    parset_file = os.path.join(DATA_DIR, "empty_parameters.parset")
    parset = Parset(parset_file)

    open_mock = mock_open()
    with patch("copperconfig.nenufar_parset.open", open_mock, create=True):
        parset.to_config_toml()

    open_mock.assert_called_with("empty_parameters.toml", "w")
    calls = [call(
        "tasks = ['process', 'rsync_quality', 'rsync']\n\n"
        "log_email = alan.loh@obspm.fr\n"
        '\n[worker]\n'
        'env_file = env_default.sh\n'
        '\n[process]\n'
        'avg_timestep = 8\n'
        'avg_freqstep = 6\n'
        'startchan = 0\n'
        'nchan = 64\n'
        'compress = False\n'
        'flag_strategy = NenuFAR-64C1S.rfis\n'
    )]

    open_mock.return_value.write.assert_has_calls(calls)
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- test_bad_avg_freq --------------------- #
# ============================================================= #
def test_bad_avg_freq():
    parset_file = os.path.join(DATA_DIR, "bad_avg_freq.parset")
    parset = Parset(parset_file)
    with pytest.raises(ValueError):
        parset.to_config_toml()
# ============================================================= #
# ============================================================= #


# ============================================================= #
# --------------------- test_bad_channels --------------------- #
# ============================================================= #
def test_bad_channels():
    parset_file = os.path.join(DATA_DIR, "bad_channels_number.parset")
    parset = Parset(parset_file)
    with pytest.raises(ValueError):
        parset.to_config_toml()
# ============================================================= #
# ============================================================= #

