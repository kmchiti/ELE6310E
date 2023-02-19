echo "---------- STARTING STEP 4 -----------"
cd ../accelergy-table-based-plug-ins/
pip3 install .
cd ../timeloop
cd src/
ln -s ../pat-public/src/pat .
cd ..
