mkdir -p timeloop-accelergy
cd timeloop-accelergy
git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
cd accelergy-timeloop-infrastructure

# "make pull" does a git clone of some modules (not sure why this is
# needed since the clone above was already recursive)
make pull
# In the past it has been necessary to checkout a specific commit of accelergy:
#cd src/accelergy
#git checkout -b python3_7 bb39de0
#cd ../..
# --End change--

cd src/cacti
make
