python3 ood/run_nero.py --in-dataset 'gastrovision' --num_classes 11 --model-arch 'resnet18'  --weights 'path/to/model/checkpoints' \
                    --seed 42     --base-dir 'path/to/save/results/file.txt' \
                    --id_path_train 'path/to/in-distribution/training/data' \
                    --id_path_valid 'path/to/in-distribution/validation/data' \
                    --ood_path 'path/to/ood/data'