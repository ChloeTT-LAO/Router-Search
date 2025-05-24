conda activate router
python src/data_synthesis/data_synthesis.py --dataset_name all --step 0
conda activate retriever
python src/data_synthesis/retriever.py --mode retrieve --dataset_name all --step 1
conda activate router
python src/data_synthesis/data_synthesis.py --dataset_name all --step 1
conda activate retriever
python src/data_synthesis/retriever.py --mode retrieve --dataset_name all --step 2
conda activate router
python src/data_synthesis/data_synthesis.py --dataset_name all --step 2
conda activate retriever
python src/data_synthesis/retriever.py --mode retrieve --dataset_name all --step 3
conda activate router
python src/data_synthesis/data_synthesis.py --dataset_name all --step 3
python src/data_synthesis/data_filter.py
python src/data_synthesis/data_combine.py