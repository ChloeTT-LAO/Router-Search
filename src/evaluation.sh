conda activate ag
python answer_generation.py --dataset_name all --method "r1-router" --model_name "r1-router" --step 0 --max_step 3
conda activate retrieve
python src/retriever.py --dataset_name all --mode retrieve --method "r1-router" --model_name "r1-router"  --step 1

conda activate ag
python answer_generation.py --dataset_name all --method "r1-router" --model_name "r1-router" --step 1 --max_step 3
conda activate retrieve
python src/retriever.py --dataset_name all --mode retrieve --method "r1-router" --model_name "r1-router"  --step 2

conda activate ag
python answer_generation.py --dataset_name all --method "r1-router" --model_name "r1-router" --step 2 --max_step 3
conda activate retrieve
python src/retriever.py --dataset_name all --mode retrieve --method "r1-router" --model_name "r1-router"  --step 3

conda activate ag
python answer_generation.py --dataset_name all --method "r1-router" --model_name "r1-router" --step 3 --max_step 3

python evaluate.py --dataset_name all --model_name "r1-router"  --method "r1-router3"