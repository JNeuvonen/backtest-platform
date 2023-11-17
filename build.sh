echo "Packaging python environment"
cd src-tauri
mkdir binaries
cd ..
cd pyserver
pyoxidizer build
cd ..
cp +r pyserver/build src-tauri/binaries/
echo "Done packaging python environment"

