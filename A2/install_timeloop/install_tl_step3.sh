cd ../accelergy
pip3 install --upgrade pip
pip3 install .
cd ../accelergy-aladdin-plug-in/
pip3 install .
cd ../accelergy-cacti-plug-in/
pip3 install .

cp -r ../cacti ~/ENV/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/

cd ../accelergy-table-based-plug-ins/
pip3 install .
