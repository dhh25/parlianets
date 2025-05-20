#!/bin/sh
cd ~
git clone https://github.com/dhh25/parlianets.git
module load tykky
conda-containerize new --prefix $INSTALL_DIR ./parlianets/environment.yml 
export PATH="$INSTALL_DIR/bin:$PATH"

