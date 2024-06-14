#!/bin/bash

# Controllare se il file train_all_pairs.sh è eseguibile
if [ ! -x train_pairs.sh ]; then
    echo "Il file train_all_pairs.sh non è eseguibile. Impostarlo come eseguibile con 'chmod +x train_pairs.sh'"
    exit 1
fi

# Eseguire train_all_pairs.sh 100 volte
for i in `seq 100`; do
    {
    echo "################# Esecuzione $i #################"
    ./train_pairs.sh
    }
done