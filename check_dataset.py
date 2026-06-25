import os

dataset_path = "dataset"

for item in os.listdir(dataset_path)[:20]:
    print(item)