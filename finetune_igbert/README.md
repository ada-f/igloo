# IglooLM and IglooALM
We show how Igloo tokens can be incorporated into a protein language model, [IgBert](https://huggingface.co/Exscientia/IgBert).

## :seedling: Setup
For the training of these models we require the following HuggingFace dependencies:
```
pip install transformers
pip install datasets
pip install accelerate
```

## :floppy_disk: Dataset
For training of inference with IglooLM and IglooALM please prepare a parquet file with the following columns:
* FW1, CDR1, FW2, CDR2, FW3, CDR4, FW4, CDR3, FW5: sequences of the sections of the chain as determined by their aho alignment. The sections can be retrieved from the aho alignment sequence with these cutoffs: 
    * light chain `FW1: (0, 23), CDR1: (23, 42), FW2: (42, 56), CDR2: (56, 72), FW3: (72, 81), CDR4: (81, 89), FW4: (89, 106), CDR3: (106, 138), FW5: (138, None)`
    * heavy chain `FW1: (0, 23), CDR1: (23, 42), FW2: (42, 56), CDR2: (56, 69), FW3: (69, 81), CDR4: (81, 89), FW4: (89, 106), CDR3: (106, 138), FW5: (138, None)`
* angles: the backbone dihedral angles, which is a numpy array of shape `[4,36,3]` for the 4 loops, dihedral angles which are zero-padded to length 36, and 3 dihedral angles $\phi, \psi, \omega$

We provide an example at `example/igloolm_training_data_sample.parquet`.

## :rocket: Inference with trained models
For an example of how the input to the model should be formatted 

### Get embeddings with IglooLM
Used for benchmarking with AbBiBench
```
python inference.py \
    --output_file embeddings.npy \
    --config_name {$PWD}/IglooLM \
    --model_name_or_path {$PWD}/IglooLM/best_step \
    --tokenizer_name {$PWD}/IglooLM/tokenizer \
    --validation_file example/igloolm_training_data_sample.parquet \
    --per_device_eval_batch_size 64 \
    --pad_to_max_length \
    --preprocessing_num_workers 16 \
    --max_seq_length 256 \
    --loop_token_model_weight checkpoints/igloo_weights.pt \
    --loop_token_model_config checkpoints/igloo_config.json \
    --use_special_cdr_tokens \
    --use_loop_tokens \
    --load_in_memory 
```

### Get embeddings with IglooALM
```
python inference.py \
    --output_file embeddings.npy \
    --config_name {$PWD}/IglooALM \
    --model_name_or_path {$PWD}/IglooALM/best_step \
    --tokenizer_name {$PWD}/IglooALM/tokenizer \
    --validation_file example/igloolm_training_data_sample.parquet \
    --per_device_eval_batch_size 64 \
    --pad_to_max_length \
    --preprocessing_num_workers 16 \
    --max_seq_length 256 \
    --loop_token_model_weight checkpoints/igloo_weights.pt \
    --loop_token_model_config checkpoints/igloo_config.json \
    --use_special_cdr_tokens \
    --use_loop_tokens_whole_sequence \
    --load_in_memory 
```

### Sample CDRs given dihedral angles with IglooALM
Sample section options are `CDR1,CDR2,CDR3,CDR4`.
```
/homefs/home/fanga5/micromamba/envs/igbert/bin/python sample_cdrs.py \
    --output_file sampled_sequences.csv \
    --config_name {$PWD}/IglooALM \
    --model_name_or_path {$PWD}/IglooALM/best_step \
    --tokenizer_name {$PWD}/IglooALM/tokenizer \
    --validation_file example/igloolm_training_data_sample.parquet \
    --loop_token_model_weight checkpoints/igloo_weights.pt \
    --loop_token_model_config checkpoints/igloo_config.json \
    --per_device_eval_batch_size 64 \
    --pad_to_max_length \
    --preprocessing_num_workers 16 \
    --max_seq_length 256 \
    --seed 42 \
    --load_in_memory \
    --use_loop_tokens_whole_sequence \
    --use_special_cdr_tokens \
    --sample_section CDR3 \
    --num_samples 10
```

## :fire: Training
We train IglooLM and IglooALM on all of pOAS with backbone dihedral angles from Ibex-predicted structures. Training of IglooLM and IglooALM uses the same datasets.

To train IglooLM which introduces the Igloo `cls` token as a special token into the protein language model please run:
```
accelerate launch finetune_igbert/train.py \
    --config_name Exscientia/IgBert \
    --model_name_or_path Exscientia/IgBert \
    --tokenizer_name Exscientia/IgBert \
    --train_file paired_OAS_index_with_loops_and_angles_single_chain_train.parquet \
    --validation_file paired_OAS_index_with_loops_and_angles_single_chain_val.parquet \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 10 \
    --output_dir igbert_igloo_model_dir \
    --pad_to_max_length \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --use_loop_tokens \
    --use_special_cdr_tokens \
    --loop_token_model_weight checkpoints/igloo_weights.pt \
    --loop_token_model_config checkpoints/igloo_config.json \
    --checkpointing_steps 2000 \
    --learning_rate 1e-5 \
    --num_warmup_steps 1000 \
    --lr_scheduler_type linear \
    --weight_decay 1e-5 \
    --report_to wandb \
    --with_tracking \
    --run_name "IglooLM"
```

To train IglooALM, please change the flag `--use_loop_tokens` to `--use_loop_tokens_whole_sequence`, which introduces the Igloo `cls` token as a special token and multimodal residue tokens for the loop residues into the protein language model.
```
accelerate launch finetune_igbert/train.py \
    --config_name Exscientia/IgBert \
    --model_name_or_path Exscientia/IgBert \
    --tokenizer_name Exscientia/IgBert \
    --train_file paired_OAS_index_with_loops_and_angles_single_chain_train.parquet \
    --validation_file paired_OAS_index_with_loops_and_angles_single_chain_val.parquet \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 10 \
    --output_dir igbert_igloo_model_dir \
    --pad_to_max_length \
    --max_seq_length 256 \
    --preprocessing_num_workers 16 \
    --use_loop_tokens_whole_sequence \
    --use_special_cdr_tokens \
    --loop_token_model_weight checkpoints/igloo_weights.pt \
    --loop_token_model_config checkpoints/igloo_config.json \
    --checkpointing_steps 2000 \
    --learning_rate 1e-5 \
    --num_warmup_steps 1000 \
    --lr_scheduler_type linear \
    --weight_decay 1e-5 \
    --report_to wandb \
    --with_tracking \
    --run_name "IglooALM"
```