for e in $(seq 0 85 3400); do
echo python runQGAN2qu.py -e ${e}
python runQGAN2qu.py -e ${e}
done