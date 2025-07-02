#!/bin/bash

# Activate virtual environment and run test
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
source venv/bin/activate
python test_pipeline.py