# MarUCOD: Unknown but Concerned Object Detection in Maritime Environments

## MarUCOD inference

```bash
python test.py \
--source ../data/images \
--name marucod_test \
--conf-thres 0.05 \
--filter-thres 0.8 \
--ood-thres 95 \
--view-img
```

## Segmentation Model (WODIS) training

```bash
python seg/train.py \
--source ../data/images \
--model_name train_wodis \
--validation \
--batch_size 4 \
--epochs 100 \
--separation_loss csol
```

## Bisecting K-means clustering

```bash
python ood/main.py
--train_data ./data/SeaShips \
--patch_size 128 \
--num_cluster 30
```

