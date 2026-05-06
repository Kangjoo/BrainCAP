source activate /gpfs/gibbs/pi/n3/software/env/pycap_env

python braincap.py --config=/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/brainCAP/brainCAP/examples/example_config.yml --steps=prep --dryrun=yes

python braincap.py --config=/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/brainCAP/brainCAP/examples/example_config.yml --steps=clustering --dryrun=yes

python braincap.py --config=/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/brainCAP/brainCAP/examples/example_config.yml --steps=prep,clustering,post --dryrun=yes

python braincap_temporal_metrics.py --analysis_folder=/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/brainCAP_tests/n100 --tag=i1 --sessions_list=/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/brainCAP/brainCAP/examples/session_list100.csv --permutations=10 --metrics="FA|mDT|vDT"

python braincap_feature_reduction.py --analysis_folder=/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/brainCAP_tests/n100 --tag=i1 --sessions_list=/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/brainCAP/brainCAP/examples/session_list100.csv

python braincap_group_comparison.py --analysis_folder=/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/brainCAP_tests/n100 --tag=i1 --sessions_list=/gpfs/gibbs/pi/n3/Studies/CAP_Time_Analytics/time-analytics/brainCAP/brainCAP/examples/session_list100.csv