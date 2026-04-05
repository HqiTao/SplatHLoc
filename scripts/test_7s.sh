export PYTHONPATH="JamMa:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_7scenes/chess \
    --cfg configs/splathloc_7scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_7scenes/fire \
    --cfg configs/splathloc_7scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_7scenes/heads \
    --cfg configs/splathloc_7scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_7scenes/office \
    --cfg configs/splathloc_7scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_7scenes/pumpkin \
    --cfg configs/splathloc_7scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_7scenes/redkitchen \
    --cfg configs/splathloc_7scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_7scenes/stairs \
    --cfg configs/splathloc_7scenes.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py


