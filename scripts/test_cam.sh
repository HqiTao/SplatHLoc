export PYTHONPATH="JamMa:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_cambridge/GreatCourt \
    --cfg configs/splathloc_cambridge.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_cambridge/KingsCollege \
    --cfg configs/splathloc_cambridge.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_cambridge/OldHospital \
    --cfg configs/splathloc_cambridge.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_cambridge/ShopFacade \
    --cfg configs/splathloc_cambridge.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py

CUDA_VISIBLE_DEVICES=0 python -W ignore splathloc.py -m map/featgs_cambridge/StMarysChurch \
    --cfg configs/splathloc_cambridge.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py
