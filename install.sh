#!/bin/bash

curl -s https://get.modular.com | sh -
python3.11 -m venv mojo-venv && source mojo-venv/bin/activate
modular install mojo
MOJO_PATH=$(modular config mojo.path) \
  && echo 'export MODULAR_HOME="'$HOME'/.modular"' >> ~/.zshrc \
  && echo 'export PATH="'$MOJO_PATH'/bin:$PATH"' >> ~/.zshrc \
  && source ~/.zshrc
pip3 install -r requirements.txt
