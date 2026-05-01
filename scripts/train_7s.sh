CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/7scenes/chess/ -m map/featgs_7scenes/chess --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/7scenes/fire/ -m map/featgs_7scenes/fire --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/7scenes/heads/ -m map/featgs_7scenes/heads --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/7scenes/office/ -m map/featgs_7scenes/office --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/7scenes/pumpkin/ -m map/featgs_7scenes/pumpkin --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/7scenes/redkitchen/ -m map/featgs_7scenes/redkitchen --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/7scenes/stairs/ -m map/featgs_7scenes/stairs --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""
