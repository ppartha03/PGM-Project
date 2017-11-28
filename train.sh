########################################################################
# Easy script to run instead of typing all model parameters every time #
########################################################################

###
# Run script for Nico's environment
###
python3 train.py --debug --gpu 1 \
    --epochs 1 \
    --batch_size 1024 \
    --optimizer 'adam' \
    --learning_rate 0.001 \
    --indep_gaussians 'f' \
    --encoder_gate 'gru' \
    --dropout_rate 0.0 \
    --embedding_size 10 \
    --encoding_size 50 \
    --encoder_layers 1 \
    --num_gaussians 10 \
    --gaussian_dim 3 \
    --decoder_layers 100 \
    --vocab_loc './data/mnist_vocab.pkl' \
    --use_mnist 'y'
    # --images_loc '../data/coco/train2014_resized256' \
    # --captions_loc '../data/coco/annotations/captions_train2014.json' \
    # --bidirectional 'y' \
    # --embeddings_loc '../data/embeddings/glove.840B.300d.txt'

###
# Add your script here and comment the others
###

