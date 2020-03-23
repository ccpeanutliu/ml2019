mkdir -p $2
python ./saliency.py $1 $2
python ./filter1.py $2
python ./filter2.py $2
python ./LIME.py $2
