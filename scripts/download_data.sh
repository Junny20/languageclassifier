# TODO: add documentation describing what this file does...

#!/bin/bash

mkdir -p data

# flags: -L -> follow redirects, -o -> output to specified file path.
curl -L -o data/wili-2018.zip https://zenodo.org/record/841984/files/wili-2018.zip
unzip data/wili-2018.zip -d data/
rm data/wili-2018.zip

echo "Done. Run scripts/preprocess.py next."