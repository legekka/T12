import os
from datasets import load_dataset as load_dataset_hf

def load_dataset(data_dir, split):
    # check if data_dir exists, if not, we will load the dataset from the hub
    if not os.path.exists(data_dir):
        return load_dataset_hf(data_dir, split=split)
    else:
        parquet_files = os.listdir(data_dir)
        parquet_files = [f for f in parquet_files if split in f]
        parquet_files = [os.path.join(data_dir, f) for f in parquet_files]

        dataset = load_dataset_hf('parquet', data_files=parquet_files, split="train")

        # check if the there's a field named labels, if there is, rename it to label
        if 'labels' in dataset.column_names:
            dataset = dataset.rename_column('labels', 'label')
            print("Renamed the column 'labels' to 'label'")

        return dataset