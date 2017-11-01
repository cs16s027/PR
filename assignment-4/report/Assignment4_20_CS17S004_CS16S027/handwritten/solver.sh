python scripts/data.py $1
python scripts/create_codebook.py $1 $2
python scripts/quantize_data.py $1 $2
python scripts/train_hmm.py $1 $2 $3
python scripts/test_hmm.py $1 $2 $3
