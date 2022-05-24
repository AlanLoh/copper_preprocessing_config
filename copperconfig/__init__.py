#! /usr/bin/python3
# -*- coding: utf-8 -*-


# Environnement file (*.sh) path 
ENV_FILE_PATH = "./"

# Flag strategy (*.rfis / *.lua) path
FLAG_STRATEGY_FILE_PATH = "./"

# Send or not Slack messages in the #alerte-nickel-preprocessing channel
SEND_SLACK_MESSAGE = True
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"

# Parameters Checks
AVAILABLE_STAT = ["SNR_XX", "SNR_YY", "RFIPercentage_XX"]

DEFAULT_ENV_FILE = "env_default.sh"

AVERAGE_TIMESTEP_MIN = 1
AVERAGE_TIMESTEP_MAX = 60
DEFAULT_AVERAGE_TIMESTEP = 8

AVERAGE_FREQSTEP_MIN = 1
DEFAULT_AVERAGE_FREQSTEP = 6

DEFAULT_STARTCHAN = 0
