python scripts/create_codebook.py $1
python scripts/quantize_data.py
python scripts/train.py $1 $2 
python scripts/test.py > results/"$1"_"$2".txt
