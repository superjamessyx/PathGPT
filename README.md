<p align="center" width="100%">
<a ><img src="src/imgs/chatpath_logo.png" alt="Llama-X" style="width: 60%; min-width: 300px; display: block; margin: auto;"></a>
</p>



[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

## ChatPath: A Knowledgeable Llama-based Chat Model for Pathology

This is a repository for ChatPath. ChatPath is a pathology-specific language model developed by fine-tuning Llama-7B on a dataset of 13k pathology-specific questions and answers that we have collected. We have released the checkpoint for ChatPath (weight diff of Llama), and we will release the 13k pathology dataset within a week. Moreover, we plan to expand the dataset to over 50,000 in the future. We believe that ChatPath will be an invaluable resource for pathologists and the pathology community.

### **Authors**

This project was completed by **Yuxuan Sun** and **Chenglu Zhu** from the Artificial Intelligence and **Biomedical Image Analysis Lab** of the School of Engineering at Westlake University. We would like to thank **Kai Zhang** (Ohio State University) for participating in the discussion and collaboration, as well as the following individuals who contributed to the annotation process: **Xinheng Lv and Ruojia Zhao**.


<h2 id="usage">Usage</h2>

- Setup. Install the conda environment:
```bash
conda create -n chatpath python=3.10
conda activate chatpath
git clone https://github.com/superjamessyx/ChatPath.git
cd ChatPath/src
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ../..
pip install -r requirements.txt
```

- Training data 

  Considering that our current data collection is restricted to pathology-related QA pairs, concentrating exclusively on QA training could impede ChatPath's efficacy in performing other tasks. To overcome this constraint, we have merged data from Stanford's [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) project with our own, yielding a diverse and extensive dataset comprising 65,000 training instances for ChatPath.
```bash
ChatPath/src/data/pathology_alpaca.json
```

- Convert LLaMA checkpoint to HuggingFace format:
```bash
cd ChatPath/src
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/llama-7B/ \
    --model_size 7B \
    --output_dir /path/to/llama-7B/hf
```

- To train LLaMA-7B with DeepSpeed, you can select either DeepSpeed Zero-2 or Zero-3 by using the following command options: `--deepspeed configs/ds_config_zero2.json` or `configs/ds_config_zero3.json`:
```bash
deepspeed train.py \
    --model_name_or_path /path/to/llama-7B/hf \
    --data_path /path/to/pathology_alpaca.json \
    --output_dir /path/to/llama-7B/hf/ft \
    --num_train_epochs 3 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/ds_config_zero2.json \
    --fp16 True
```
- Train LLaMA-7B on DeepSpeed with Multi-nodes
```bash
deepspeed --num_gpus num_of_gpus_in_each_node \
    --num_nodes num_of_nodes \
    --master_addr ip_address_of_main_node \
    --master_port 34545 \
    --hostfile configs/hostfile \
    train.py \
    --model_name_or_path /path/to/llama-7B/hf \
    --data_path /path/to/pathology_alpaca.json \
    --output_dir /path/to/llama-7B/hf/ft \
    --num_train_epochs 3 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/ds_config_zero2.json \
    --fp16 True
```



- Training Cost

Training 7B model with `13k examples/ 3 epoch` & `batch_size=64` within `1 hour` on `4 x A100 GPUs`.

| LLaMA | Batch Size | A100s | Time (h) |
| ----- | ---------- | ----- | -------- |
| 7 B   | 32         | 4     | 4.50     |


- Inference
```bash
# web demo inference
python generate.py
```




## Thanks For

This project is based on 

[Llama-X](https://github.com/AetherCortex/Llama-X)

and also relate to multiple open source projects:

[Meta AI LLaMA](https://arxiv.org/abs/2302.13971v1)

[Huggingface Transformers Llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama)

[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)


## Disclaimer

The use of resources(e.g., code, data and model weights) related to this project is limited to academic research and is prohibited for commercial purposes. The content generated by ChatPath is subject to factors such as randomness and uncontrollability, and this project cannot guarantee its accuracy. This project does not assume any legal responsibility for the content of the model output, nor does it assume any responsibility for any losses that may arise from the use of related resources and output results.



