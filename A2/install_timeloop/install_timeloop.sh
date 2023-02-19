# Installation script for Timeloop-Accelergy on Ubuntu and Google Colab
# Author: Francois Leduc-Primeau <francois.leduc-primeau@polymtl.ca>

# Usage: source <SCRIPT LOCATION>/install_timeloop.sh
# where <SCRIPT LOCATION> can be "." if your current
# directory is the location of this script.

# NOTE: On Google Colab, you probably want to access this script from
# a location in your google drive. Your google drive can be mounted by
# executing the following notebook cell:
### from google.colab import drive
### drive.mount('/content/drive')

# ---------------------------------------------------------------------

# Get location of this script
# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
# Create a symlink in the home directory
ln -s ${DIR} ~/install_tl

cd ~
source ~/install_tl/install_tl_step0.sh

source ~/install_tl/install_tl_step1.sh
# we should now be in a python virtual environment

source ~/install_tl/install_tl_step2.sh

source ~/install_tl/install_tl_step3.sh

source ~/install_tl/install_tl_step4.sh

source ~/install_tl/install_tl_step4b.sh

source ~/install_tl/install_tl_step5.sh
