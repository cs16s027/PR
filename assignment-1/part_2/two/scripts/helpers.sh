ffmpeg -framerate 3 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p trend.mp4

# Regression script for 2nd order, 100 epochs, plain
python scripts/regression.py data/20_2.txt 0.0 2 100 models/plain/2/m100.npy False False plots/plain/2/p0100.png

# Regression script for 3nd order, 100 epochs, plain
python scripts/regression.py data/20_2.txt 0.0 3 100 models/plain/3/m100.npy False False plots/plain/3/p0100.png

# Regression script for 3nd order, 100 epochs, plain
python scripts/regression.py data/20_2.txt 0.001 3 100 models/plain/3/m100.npy False False plots/plain/3/p0100.png

