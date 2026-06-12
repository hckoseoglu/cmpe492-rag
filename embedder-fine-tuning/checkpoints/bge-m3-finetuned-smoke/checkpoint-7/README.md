---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:50
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-base-en-v1.5
widget:
- source_sentence: what are the two attachments of a limb muscle to bone called?
  sentences:
  - The functions of the endocrine system, nervous system, immune system, and musculoskeletal
    system are integrated. Communication among systems is accomplished with hormones
    and other peptides. Strength and conditioning professionals and athletes appreciate
    the importance of anabolic hormones for mediating changes in the body and helping
    with the adaptive response to heavy resistance training. The goal of this chapter
    is to provide an initial glimpse into the endocrine system that mediates the changes
    in the body with training.
  - 'Each skeletal muscle is an organ that contains muscle tissue, connective tissue,
    nerves, and blood vessels. Fibrous connective tissue, or epimysium, covers the
    body’s more than 430 skeletal muscles. The epimysium is continuous with the tendons
    at the ends of the muscle. The tendon is attached to bone periosteum, a specialized
    connective tissue covering all bones; any contraction of the muscle pulls on the
    tendon and, in turn, the bone. Limb muscles have two attachments to bone: proximal
    (closer to the trunk) and distal (farther from the trunk). Trunk muscles have
    two attachments termed superior (closer to the head) and inferior (closer to the
    feet). Traditionally, the origin of a muscle is defined as its proximal (toward
    the center of the body) attachment, and the insertion is defined as its distal
    (away from the center of the body) attachment.'
  - The fibrous connective tissues that surround and separate the different organizational
    levels within skeletal muscle are referred to as fascia. Fascia have sheets of
    fibrocollagenous support tissue containing bundles of collagen fibers arranged
    in different planes to provide resistance to forces from different directions.
    Fascia within muscles converge together near the end of the muscle to form a tendon
    through which the force of muscle contraction is transmitted to bone.
- source_sentence: What are the essential anatomical and functional considerations
    of the neuromuscular, respiratory, and cardiovascular systems for developing effective
    training programs?
  sentences:
  - In order to best apply the available scientific knowledge to the training of athletes
    and the development of effective training programs, strength and conditioning
    professionals must have a basic understanding of not only skeletal muscle function
    but also those systems of the body that directly support the work of exercising
    muscle. Accordingly, this chapter summarizes those aspects of the anatomy and
    function of the neuromuscular, respiratory, and cardiovascular systems that are
    essential for developing and maintaining muscular force and power.
  - In order to best apply the available scientific knowledge to the training of athletes
    and the development of effective training programs, strength and conditioning
    professionals must have a basic understanding of not only skeletal muscle function
    but also those systems of the body that directly support the work of exercising
    muscle. Accordingly, this chapter summarizes those aspects of the anatomy and
    function of the neuromuscular, respiratory, and cardiovascular systems that are
    essential for developing and maintaining muscular force and power.
  - Knowledge of muscular, neuromuscular, cardiovascular, and respiratory anatomy
    and physiology is important for the strength and conditioning professional to
    have in order to understand the scientific basis for conditioning. This includes
    knowledge of the function of the macrostructure and microstructure of muscle fibers,
    muscle fiber types, interactions between tendon and muscle and between the motor
    unit and its activation, as well as the interactions of the heart, vascular system,
    lungs, and respiratory system.
- source_sentence: What are the primary mechanisms by which different bodily systems,
    such as the endocrine and nervous systems, communicate with each other?
  sentences:
  - The functions of the endocrine system, nervous system, immune system, and musculoskeletal
    system are integrated. Communication among systems is accomplished with hormones
    and other peptides. Strength and conditioning professionals and athletes appreciate
    the importance of anabolic hormones for mediating changes in the body and helping
    with the adaptive response to heavy resistance training. The goal of this chapter
    is to provide an initial glimpse into the endocrine system that mediates the changes
    in the body with training. Knowledge of musculoskeletal anatomy and biomechanics
    is important for understanding human movements, including those involved in sport
    and resistance exercise.
  - 'The functions of the endocrine system, nervous system, immune system, and musculoskeletal
    system are integrated.

    Communication among systems is accomplished with hormones and other peptides.'
  - Finally, by way of the endocrine system, the brain influences the various endocrine
    glands of the body, which release hormones such as testosterone, cortisol, and
    thyroxin that can dramatically affect the physiological state via anabolic and
    catabolic processes.
- source_sentence: What three primary areas of physical performance does a strength
    and conditioning professional aim to enhance?
  sentences:
  - The primary goals of a strength and conditioning program are to lower the potential
    for injury and improve performance.
  - In order to best apply the available scientific knowledge to the training of athletes
    and the development of effective training programs, strength and conditioning
    professionals must have a basic understanding of not only skeletal muscle function
    but also those systems of the body that directly support the work of exercising
    muscle. Accordingly, this chapter summarizes those aspects of the anatomy and
    function of the neuromuscular, respiratory, and cardiovascular systems that are
    essential for developing and maintaining muscular force and power.
  - At the most basic level, the strength and conditioning professional is concerned
    with maximizing physical performance and must therefore conduct programs that
    are designed to increase muscular strength, muscular endurance, and flexibility.
    However, in addition to being concerned about the function and control of muscle
    through the motor unit (the basic functional unit of the human neuromuscular system),
    the professional must be cognizant of how the cardiovascular and respiratory systems
    interact with the neuromuscular system to provide an optimal environment for sustaining
    muscular work.
- source_sentence: What are the primary mechanisms by which different bodily systems,
    such as the endocrine and nervous systems, communicate with each other?
  sentences:
  - 'The functions of the endocrine system, nervous system, immune system, and musculoskeletal
    system are integrated.

    Communication among systems is accomplished with hormones and other peptides.'
  - The inflammatory process involves the immune system and various immune cells (e.g.,
    T cells), which are under endocrine control. The study of the connection between
    the neural, endocrine, and immune systems is called neuroendocrine immunology.
  - The functions of the endocrine system, nervous system, immune system, and musculoskeletal
    system are integrated. Communication among systems is accomplished with hormones
    and other peptides. Strength and conditioning professionals and athletes appreciate
    the importance of anabolic hormones for mediating changes in the body and helping
    with the adaptive response to heavy resistance training. The goal of this chapter
    is to provide an initial glimpse into the endocrine system that mediates the changes
    in the body with training. Knowledge of musculoskeletal anatomy and biomechanics
    is important for understanding human movements, including those involved in sport
    and resistance exercise.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-base-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) <!-- at revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'cls', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'What are the primary mechanisms by which different bodily systems, such as the endocrine and nervous systems, communicate with each other?',
    'The functions of the endocrine system, nervous system, immune system, and musculoskeletal system are integrated. Communication among systems is accomplished with hormones and other peptides. Strength and conditioning professionals and athletes appreciate the importance of anabolic hormones for mediating changes in the body and helping with the adaptive response to heavy resistance training. The goal of this chapter is to provide an initial glimpse into the endocrine system that mediates the changes in the body with training. Knowledge of musculoskeletal anatomy and biomechanics is important for understanding human movements, including those involved in sport and resistance exercise.',
    'The inflammatory process involves the immune system and various immune cells (e.g., T cells), which are under endocrine control. The study of the connection between the neural, endocrine, and immune systems is called neuroendocrine immunology.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7498, 0.6679],
#         [0.7498, 1.0000, 0.6974],
#         [0.6679, 0.6974, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 50 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 50 samples:
  |         | anchor                                                                             | positive                                                                             | negative                                                                            |
  |:--------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                               | string                                                                              |
  | details | <ul><li>min: 14 tokens</li><li>mean: 22.56 tokens</li><li>max: 30 tokens</li></ul> | <ul><li>min: 40 tokens</li><li>mean: 109.98 tokens</li><li>max: 197 tokens</li></ul> | <ul><li>min: 22 tokens</li><li>mean: 94.52 tokens</li><li>max: 350 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                  | positive                                                                                                                                                                                                       | negative                                                                                                                                                                                                                                                                                                                                                                                                            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What are the primary mechanisms by which different bodily systems, such as the endocrine and nervous systems, communicate with each other?</code> | <code>The functions of the endocrine system, nervous system, immune system, and musculoskeletal system are integrated.<br>Communication among systems is accomplished with hormones and other peptides.</code> | <code>The inflammatory process involves the immune system and various immune cells (e.g., T cells), which are under endocrine control. The study of the connection between the neural, endocrine, and immune systems is called neuroendocrine immunology.</code>                                                                                                                                                    |
  | <code>What are the primary mechanisms by which different bodily systems, such as the endocrine and nervous systems, communicate with each other?</code> | <code>The functions of the endocrine system, nervous system, immune system, and musculoskeletal system are integrated.<br>Communication among systems is accomplished with hormones and other peptides.</code> | <code>Finally, by way of the endocrine system, the brain influences the various endocrine glands of the body, which release hormones such as testosterone, cortisol, and thyroxin that can dramatically affect the physiological state via anabolic and catabolic processes.</code>                                                                                                                                 |
  | <code>What are the primary mechanisms by which different bodily systems, such as the endocrine and nervous systems, communicate with each other?</code> | <code>The functions of the endocrine system, nervous system, immune system, and musculoskeletal system are integrated.<br>Communication among systems is accomplished with hormones and other peptides.</code> | <code>There are three pathways leading from the brain and spinal cord (the central nervous system (CNS)) to the athlete’s physical apparatus (bone, muscle, nerves, vasculature, and glands). First, there are connections via the voluntary nervous outflow (i.e., pyramidal and extrapyramidal systems) to the skeletal muscles. The cerebral cortex, where thought occurs, is ‘hardwired’ to the muscles.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 1
- `learning_rate`: 2e-05
- `warmup_steps`: 0.1
- `gradient_checkpointing`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 2e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.1429 | 1    | 1.0772        |
| 0.2857 | 2    | 1.0253        |
| 0.4286 | 3    | 0.7817        |
| 0.5714 | 4    | 0.6592        |
| 0.7143 | 5    | 0.6512        |
| 0.8571 | 6    | 0.4225        |
| 1.0    | 7    | 0.8013        |


### Training Time
- **Training**: 15.2 seconds

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.4.1
- Transformers: 5.7.0
- PyTorch: 2.11.0
- Accelerate: 1.13.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->