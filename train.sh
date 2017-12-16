########################################################################
# Easy script to run instead of typing all model parameters every time #
########################################################################

###
# Run script for Nico's environment
###
python3 train.py --gpu 0 \
    --save_prefix 'models/indep/VED_mnist-op' \
    --classifier_prefix 'models/1513461532.108166' \
    --epochs 1000 \
    --batch_size 128 \
    --optimizer 'adam' \
    --learning_rate 0.001 \
    --indep_gaussians 't' \
    --encoder_gate 'gru' \
    --dropout_rate 0.0 \
    --fix_embeddings 't' \
    --encoding_size 30 \
    --encoder_layers 1 \
    --num_gaussians 10 \
    --gaussian_dim 30 \
    --decoder_layers 500 \
    --vocab_loc './data/mnist_vocab.pkl' \
    --use_mnist 't' \
    --distractors 100 \
    --bidirectional 'f' \
    # --images_loc '../data/coco/train2014_resized256' \
    # --captions_loc '../data/coco/annotations/captions_train2014.json' \
    # --embeddings_loc '../data/embeddings/glove.840B.300d.txt' \

###
# Add your script here and comment the others
###

