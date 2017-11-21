########################################################################
# Easy script to run instead of typing all model parameters every time #
########################################################################

# Run script for Nico's environment
python3 train.py --debug \
    --epochs 1 \
    --learning_rate 0.001 \
    --num_gaussians 2 \
    --gaussian_dim 16 \
    --images_loc '../data/coco/train2014_resized256' \
    --captions_loc '../data/coco/annotations/captions_train2014.json' \
    --vocab_loc './data/vocab-t4_09956.pkl'

# Add your script here and comment the others

