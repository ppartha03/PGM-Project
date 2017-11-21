########################################################################
# Easy script to run instead of typing all model parameters every time #
########################################################################

# Run script for Nico's environment
python3 train.py --debug \
    --epochs 1 \
    --batch_size 1024 \
    --optimizer 'adam' \
    --learning_rate 0.001 \
    --num_gaussians 2 \
    --gaussian_dim 16 \
    --encoding_size 500 \
    --encoder_layers 1 \
    --embeddings_loc '../data/embeddings/glove.840B.300d.txt' \
    --images_loc '../data/coco/train2014_resized256' \
    --captions_loc '../data/coco/annotations/captions_train2014.json' \
    --vocab_loc './data/vocab-t4_09956.pkl'

# Add your script here and comment the others
