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
    
def filter_dataset(dataset, num_labels, minimum_labels=-1, num_proc=12):

    def _keep_labels(examples):
        batch_labels = []
        for example in examples["label"]:
            labels = []
            for label in example:
                if label < num_labels:
                    labels.append(label)
            batch_labels.append(labels)
        examples["label"] = batch_labels
        return examples
    
    def _keep_minimum_labels(examples):
        mask = []
        for example in examples["label"]:
            if len(example) >= minimum_labels:
                mask.append(True)
            else:
                mask.append(False)
        return mask
    
    # Changing example["label"] to only include labels that are less than num_labels
    dataset = dataset.map(_keep_labels, batched=True, num_proc=num_proc, desc="Selecting labels less than num_labels")

    # Removing examples that have less than minimum_labels
    if minimum_labels > 0:
        dataset = dataset.filter(_keep_minimum_labels, batched=True, num_proc=num_proc, desc="Filtering examples with less than minimum_labels")

    return dataset