#!/usr/bin/env bash

if [ ! -d "venv" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  python3 -m venv venv
fi

/bin/zsh  -c ". ./venv/bin/activate ; exec pip install -r requirements.txt"
/bin/zsh  -c ". ./venv/bin/activate ; exec jupyter notebook --port 8888"

