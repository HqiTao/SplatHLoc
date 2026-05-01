CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/apt1/kitchen/ -m map/featgs_12scenes/apt1_kitchen --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/apt1/living/ -m map/featgs_12scenes/apt1_living --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/apt2/bed/ -m map/featgs_12scenes/apt2_bed --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/apt2/kitchen/ -m map/featgs_12scenes/apt2_kitchen --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/apt2/living/ -m map/featgs_12scenes/apt2_living --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/apt2/luke/ -m map/featgs_12scenes/apt2_luke --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/office1/gates362/ -m map/featgs_12scenes/gates362 --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/office1/gates381/ -m map/featgs_12scenes/gates381 --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/office1/lounge/ -m map/featgs_12scenes/lounge --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/office1/manolis/ -m map/featgs_12scenes/manolis --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/office2/5a/ -m map/featgs_12scenes/5a --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""

CUDA_VISIBLE_DEVICES=0 python train_featgs.py -s datasets/12scenes/office2/5b/ -m map/featgs_12scenes/5b --iterations 30000 \
    --data_device cpu -f sp -g featgs --images  ""
