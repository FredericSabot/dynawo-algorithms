#!/bin/bash
nbThreads=4

for i in {0..49}
do
    python N-2Analysis.py --working_dir /home/fsabot/Desktop/MyDynawoFiles/IEEE39/IEEE39_Auto_Protections/protected$i --name IEEE39
    echo "Generated files for batch $i"
    /home/fsabot/Desktop/dynawo-algorithms/myEnvDynawoAlgorithms.sh SA --directory /home/fsabot/Desktop/MyDynawoFiles/IEEE39/IEEE39_Auto_Protections/protected$i/N-2_Analysis --input fic_MULTIPLE.xml --output aggregatedResults.xml --nbThreads $nbThreads
    echo "Batch $i completed"
done
