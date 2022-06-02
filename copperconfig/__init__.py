#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os

# Environnement file (*.sh) path 
ENV_FILE_PATH = "./"

# Flag strategy (*.rfis / *.lua) path
FLAG_STRATEGY_FILE_PATH = "./"
DEFAULT_FLAG_RFI = "true"
DEFAULT_FLAG_MEMORYPERC = 30 # not set via parameters

# Send or not Slack messages in the #alerte-nickel-preprocessing channel
SEND_SLACK_MESSAGE = True
SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL')

# Parameters Checks
AVAILABLE_STAT = ["SNR_XX", "SNR_YY", "RFIPercentage_XX"]

DEFAULT_ENV_FILE = "env_default.sh"

AVERAGE_TIMESTEP_MIN = 1
AVERAGE_TIMESTEP_MAX = 60
DEFAULT_AVERAGE_TIMESTEP = 8

AVERAGE_FREQSTEP_MIN = 1
DEFAULT_AVERAGE_FREQSTEP = 6

DEFAULT_STARTCHAN = 0
