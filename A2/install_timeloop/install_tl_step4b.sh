# this step took a few minutes for me
scons -j8 --accelergy --static
# copy the executables in the path
# Note: Environment variable TL_INSTALL_PREFIX is created in the
# top-level script.
mkdir -p ${TL_INSTALL_PREFIX}/bin
cp -v build/timeloop-* ${TL_INSTALL_PREFIX}/bin/
