export PYTHONPATH="JamMa:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/apt1_kitchen \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/apt1_living \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/apt2_bed \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/apt2_kitchen \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/apt2_living \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/apt2_luke \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/gates362 \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/gates381 \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/lounge \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/manolis \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/5a \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_12scenes/5b \
    --cfg configs/splathloc_12scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py
