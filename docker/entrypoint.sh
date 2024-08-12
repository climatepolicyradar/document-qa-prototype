#!/usr/bin/env bash
# This is the default entrypoint for the official Prefect Docker image
# Modified to use poetry for dependency management; assuming multi-stage build that adds poetry libs to the image

set -e

if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi

if [ ! -z "$EXTRA_PIP_PACKAGES" ]; then
  echo "+pip install $EXTRA_PIP_PACKAGES"
  pip install $EXTRA_PIP_PACKAGES
fi

if [ -z "$*" ]; then
  echo "\

______          __          _         __  _____ ____________ 
| ___ \        / _|        | |       / / /  __ \| ___ \ ___ \
| |_/ / __ ___| |_ ___  ___| |_     / /  | /  \/| |_/ / |_/ /
|  __/ '__/ _ \  _/ _ \/ __| __|   / /   | |    |  __/|    / 
| |  | | |  __/ ||  __/ (__| |_   / /    | \__/\| |   | |\ \ 
\_|  |_|  \___|_| \___|\___|\__| /_/      \____/\_|   \_| \_|
                                                             
                                                             
"
  exec bash --login
else
  if [ "$1" = "python" ]; then
    shift
    exec poetry run python "$@"
  else
    exec "$@"
  fi
fi
