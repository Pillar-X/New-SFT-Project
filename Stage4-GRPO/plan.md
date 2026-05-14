说明：
基础模型在/root/New-SFT-Project/Stage2-LoRA-Finetune/models/Qwen3-0.6B-Base
lora权重在/root/New-SFT-Project/Stage3-Evaluation/lora，现在先默认为：lora2-500steps
训练数据集在/root/New-SFT-Project/Stage2-LoRA-Finetune/data/sft，先默认用sft_boxed_small.json

为了进一步提升，请你设计GRPO算法，利用数据集，继续训练lora权重。