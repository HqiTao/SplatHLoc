# Convert nvm model to colmap model

python datasets/colmap_from_nvm.py --nvm_path datasets/cambridge/GreatCourt/reconstruction.nvm --colmap_path datasets/cambridge/GreatCourt/sparse/0

python datasets/colmap_from_nvm.py --nvm_path datasets/cambridge/KingsCollege/reconstruction.nvm --colmap_path datasets/cambridge/KingsCollege/sparse/0

python datasets/colmap_from_nvm.py --nvm_path datasets/cambridge/OldHospital/reconstruction.nvm --colmap_path datasets/cambridge/OldHospital/sparse/0

python datasets/colmap_from_nvm.py --nvm_path datasets/cambridge/ShopFacade/reconstruction.nvm --colmap_path datasets/cambridge/ShopFacade/sparse/0

python datasets/colmap_from_nvm.py --nvm_path datasets/cambridge/StMarysChurch/reconstruction.nvm --colmap_path datasets/cambridge/StMarysChurch/sparse/0

# preprocess
python datasets/preprocess.py --source_path datasets/cambridge/GreatCourt --output_folder processed

python datasets/preprocess.py --source_path datasets/cambridge/KingsCollege --output_folder processed

python datasets/preprocess.py --source_path datasets/cambridge/OldHospital --output_folder processed

python datasets/preprocess.py --source_path datasets/cambridge/ShopFacade --output_folder processed

python datasets/preprocess.py --source_path datasets/cambridge/StMarysChurch --output_folder processed