CUDA_VISIBLE_DEVICES=0 python train_featgs.py  -s datasets/cambridge/GreatCourt -m map/featgs_cambridge/GreatCourt -r 1  -f sp -g featgs --iterations 30000 \
    --data_device cpu --train_detector --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001 

CUDA_VISIBLE_DEVICES=0 python train_featgs.py  -s datasets/cambridge/KingsCollege -m map/featgs_cambridge/KingsCollege -r 1  -f sp -g featgs --iterations 30000 \
    --data_device cpu --train_detector --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001

CUDA_VISIBLE_DEVICES=0 python train_featgs.py  -s datasets/cambridge/OldHospital -m map/featgs_cambridge/OldHospital -r 1  -f sp -g featgs --iterations 30000 \
    --data_device cpu --train_detector --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001

CUDA_VISIBLE_DEVICES=0 python train_featgs.py  -s datasets/cambridge/ShopFacade -m map/featgs_cambridge/ShopFacade -r 1  -f sp -g featgs --iterations 30000 \
    --data_device cpu --train_detector --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001

CUDA_VISIBLE_DEVICES=0 python train_featgs.py  -s datasets/cambridge/StMarysChurch -m map/featgs_cambridge/StMarysChurch -r 1  -f sp -g featgs --iterations 30000 \
    --data_device cpu --train_detector --densify_grad_threshold 0.0004 --images "processed" --position_lr_init 0.000016 --scaling_lr 0.001
