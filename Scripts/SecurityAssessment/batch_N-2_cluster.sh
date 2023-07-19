#!/bin/bash
# Submission script for Lemaitre3
#SBATCH --job-name=Dynawo
#SBATCH --array=0-49
#SBATCH --time=01:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000 # megabytes
#SBATCH --partition=batch,debug
#
#SBATCH --mail-user=frederic.sabot@ulb.be
#SBATCH --mail-type=BEGIN, END, FAIL, INVALID_DEPEND, REQUEUE, STAGE_OUT, TIME_LIMIT_90

module purge
module load GCC Python

# Run once manually before launching the runs
# python scripts/Protections.py --working_dir IEEE39 --name IEEE39 --output protected --randomise

echo "Task ID: $SLURM_ARRAY_TASK_ID"
python scripts/N-2Analysis.py --working_dir IEEE39/protected$SLURM_ARRAY_TASK_ID --name IEEE39
echo "Generated files for batch $SLURM_ARRAY_TASK_ID"
./dynawo-algorithms_no_mpi/dynawo-algorithms.sh SA --directory IEEE39/protected$SLURM_ARRAY_TASK_ID/N-2_Analysis --input fic_MULTIPLE.xml --output aggregatedResults.xml # --nbThreads 10

python scripts/AnalyseTimelines.py --working_dir IEEE39/protected$SLURM_ARRAY_TASK_ID/N-2_Analysis --name IEEE39
cp IEEE39/protected$SLURM_ARRAY_TASK_ID/N-2_Analysis/TimelineAnalysis.csv IEEE39/MC_results/TimelineAnalysis$SLURM_ARRAY_TASK_ID.csv
echo "Batch $SLURM_ARRAY_TASK_ID completed"

# Run once manually after all runs
# python scripts/MergeTimeLineAnalyses.py --working_dir IEEE39/MC_results
