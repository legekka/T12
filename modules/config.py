import json

class Config:
    def __init__(self, config_path=None, jsonData=None):
        if config_path is not None:
            with open(config_path, 'r', encoding="utf-8") as f:
                jsonData = json.load(f)
            self._jsonData = jsonData
        elif jsonData is not None:
            self._jsonData = jsonData
        else:
            raise ValueError("Either config_path or jsonData must be provided.")
        
        self.load()
        
    def load(self):
        self.name = self._jsonData["name"]
        self.model = self._jsonData["model"]
        
        self.output_dir = self._jsonData["output_dir"]

        if isinstance(self._jsonData["train_dataset"], str):
            self.train_dataset = self._jsonData["train_dataset"]
        elif isinstance(self._jsonData["train_dataset"], dict):
            self.train_dataset = {
                "path": self._jsonData["train_dataset"]["path"],
                "name": self._jsonData["train_dataset"]["name"],
                "split": self._jsonData["train_dataset"]["split"],
            }
        else:
            raise ValueError("train_dataset must be a string or a dictionary.")

        self.eval_dataset_split_size = self._jsonData["eval_dataset_split_size"] if "eval_dataset_split_size" in self._jsonData else 1000

        self.max_seq_length = self._jsonData["max_seq_length"] if "max_seq_length" in self._jsonData else None
        self.num_classes = self._jsonData["num_classes"] if "num_classes" in self._jsonData else None
        self.train_classes = self._jsonData["train_classes"] if "train_classes" in self._jsonData else self.num_classes
        self.minimum_classes= self._jsonData["minimum_classes"] if "minimum_classes" in self._jsonData else -1

        self.batch_size = self._jsonData["batch_size"] if "batch_size" in self._jsonData else 8
        self.gradient_accumulation_steps = self._jsonData["gradient_accumulation_steps"] if "gradient_accumulation_steps" in self._jsonData else 1
        self.num_workers = self._jsonData["num_workers"] if "num_workers" in self._jsonData else 0
        self.num_epochs = self._jsonData["num_epochs"] if "num_epochs" in self._jsonData else None
        self.max_steps = self._jsonData["max_steps"] if "max_steps" in self._jsonData else None

        self.learning_rate = self._jsonData["learning_rate"]
        self.d_coef = self._jsonData["d_coef"] if "d_coef" in self._jsonData else 1
        self.eta_min = self._jsonData["eta_min"] if "eta_min" in self._jsonData else 0.0
        self.warmup_steps = self._jsonData["warmup_steps"] if "warmup_steps" in self._jsonData else 0

        self.eval_steps = self._jsonData["eval_steps"] if "eval_steps" in self._jsonData else None
        self.save_steps = self._jsonData["save_steps"] if "save_steps" in self._jsonData else None
        self.logging_steps = self._jsonData["logging_steps"] if "logging_steps" in self._jsonData else 5    
        self.mlm_probability = self._jsonData["mlm_probability"] if "mlm_probability" in self._jsonData else 0.15


        self.criterion = self._jsonData["criterion"] if "criterion" in self._jsonData else "CrossEntropyLoss"
        self.optimizer = self._jsonData["optimizer"] if "optimizer" in self._jsonData else "AdamW"
        self.scheduler = self._jsonData["scheduler"] if "scheduler" in self._jsonData else "WarmupThenCosineAnnealingLR"

        if "wandb" in self._jsonData:
            self.wandb = {
                "project": self._jsonData["wandb"]["project"],
                "name": self._jsonData["wandb"]["name"] if "name" in self._jsonData["wandb"] else None,
                "tags": self._jsonData["wandb"]["tags"] if "tags" in self._jsonData["wandb"] else None
            }
        
    def save(self, path):
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(self._jsonData, f, ensure_ascii=False, indent=4)