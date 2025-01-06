# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/PP-OCRv3/multi_language/devanagari_PP-OCRv3_rec.yml
