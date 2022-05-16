#! /usr/bin/python3
# -*- coding: utf-8 -*-


__author__ = 'Alan Loh'
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'


import pytest
import os
from copperconfig.nenufar_parset import Parset


# ============================================================= #
# ---------------- test_fixedtarget_properties ---------------- #
# ============================================================= #
def test_parset_reading():
    parset_file = os.path.join(os.path.dirname(__file__), "parsets/sun.parset")
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

    parset.to_config_tml()
# ============================================================= #
# ============================================================= #