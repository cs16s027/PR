for i in `seq 5 5 250`;do 
    python scripts/basicEVD.py data/20_square.jpg "$i" reconstructions/True/20_square_N_"$i".jpg plots/20_square_plot.jpg T;
done
