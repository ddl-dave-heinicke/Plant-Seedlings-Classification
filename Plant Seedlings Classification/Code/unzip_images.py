import zipfile
with zipfile.ZipFile('/mnt/data/Plant-Seedlings-Classification/plant-seedlings-classification.zip', 'r') as zip_ref:
    zip_ref.extractall('/mnt/data/Plant-Seedlings-Classification/plant-seedlings-classification')