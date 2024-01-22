echo "Packaging python environment"
npx kill-port 8000
rm -rf src-tauri/binaries/
rm -rf pyserver/build/
cd src-tauri
mkdir binaries
cd ..
cd pyserver
pyoxidizer build
cd ..
cp -r pyserver/build src-tauri/binaries/
echo "Done packaging python environment"
