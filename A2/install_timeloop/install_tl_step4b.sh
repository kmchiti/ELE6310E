echo "---------- STARTING STEP 4b -----------"
# this step took a few minutes for me
scons -j8 --accelergy --static
cp build/timeloop-* /bin/
