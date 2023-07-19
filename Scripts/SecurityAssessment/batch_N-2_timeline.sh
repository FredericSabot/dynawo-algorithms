#!/bin/bash

mkdir -p /home/fsabot/Desktop/MyDynawoFiles/IEEE39/IEEE39_Auto_Protections/MC_results
for i in {0..49}
do
    python3 AnalyseTimelines.py --working_dir /home/fsabot/Desktop/MyDynawoFiles/IEEE39/IEEE39_Auto_Protections/protected$i/N-2_Analysis --name IEEE39
    cp /home/fsabot/Desktop/MyDynawoFiles/IEEE39/IEEE39_Auto_Protections/protected$i/N-2_Analysis/TimelineAnalysis.csv /home/fsabot/Desktop/MyDynawoFiles/IEEE39/IEEE39_Auto_Protections/MC_results/TimelineAnalysis$i.csv
    echo "Batch $i completed"
done

python MergeTimeLineAnalyses.py --working_dir /home/fsabot/Desktop/MyDynawoFiles/IEEE39/IEEE39_Auto_Protections/MC_results
