import torch.nn as nn
from transformers import LongformerModel

class DifficultyClassifier(nn.Module):
    def __init__(self, n_classes, model_name='allenai/longformer-base-4096'):
        super(DifficultyClassifier, self).__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.longformer.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)