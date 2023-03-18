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
# CONFIG

## Timeloop installation path
#export TL_INSTALL_PREFIX = "/bin" # for Google Colab
export TL_INSTALL_PREFIX="${HOME}/.local" # for other cases (adjust as necessary)

## Whether to copy previously saved timeloop executables rather than
## recompiling
#export TL_USE_SAVED_TIMELOOP=1
## Location where executables can be saved
export GOOGLE_DRIVE_PATH="/content/drive/MyDrive"
export TL_EXEC_SAVE_PATH=${GOOGLE_DRIVE_PATH}/timeloop_colab_executables
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
test -e ~/install_tl && rm -i ~/install_tl
echo "Creating symlink to ${DIR}"
ln -sv ${DIR} ~/install_tl

cd ~
source ~/install_tl/install_tl_step0.sh

echo "---------- STARTING STEP 1 -----------"
source ~/install_tl/install_tl_step1.sh
# we should now be in a python virtual environment

echo "---------- STARTING STEP 2 -----------"
source ~/install_tl/install_tl_step2.sh

echo "---------- STARTING STEP 3 -----------"
source ~/install_tl/install_tl_step3.sh

# Compile and install Timeloop
echo "---------- STARTING STEP 4 -----------"
if [ $TL_USE_SAVED_TIMELOOP -eq 1 ]
then
		echo "Installing previously saved Timeloop executables:";
		mkdir -p ${TL_INSTALL_PREFIX}/bin;
		cp -v ${TL_EXEC_SAVE_PATH}/timeloop-* ${TL_INSTALL_PREFIX}/bin/;
		chmod u+x ${TL_INSTALL_PREFIX}/bin/*
else
		echo "Compiling Timeloop...";
		source ~/install_tl/install_tl_step4.sh;
		echo "---------- STARTING STEP 4b -----------";
		source ~/install_tl/install_tl_step4b.sh
fi

echo "---------- STARTING STEP 5 -----------"
source ~/install_tl/install_tl_step5.sh
