---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:26300
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: How does the accumulation of lactate during exercise contribute
    to muscle fatigue?
  sentences:
  - The formation of lactate from pyruvate is catalyzed by the enzyme lactate dehydrogenase.
    At physiological pH (i.e., near 7), the lactic acid molecule cannot exist. Lactate,
    not lactic acid, is the result of the lactate dehydrogenase reaction. Although
    the muscular fatigue experienced during exercise often correlates with high tissue
    concentrations of lactate, lactate is not the cause of fatigue. The H+ accumulation
    as a result of lactate formation reduces the intracellular pH, inhibits glycolytic
    reactions, and directly interferes with muscle’s excitation-contraction coupling.
  - The formation of lactate from pyruvate is catalyzed by the enzyme lactate dehydrogenase.
    Lactate, not lactic acid, is the result of the lactate dehydrogenase reaction.
    Although the muscular fatigue experienced during exercise often correlates with
    high tissue concentrations of lactate, lactate is not the cause of fatigue.
  - Along with exercise intensity and muscle fiber type, exercise duration, state
    of training, and initial glycogen levels can also influence lactate accumulation.
- source_sentence: Which macronutrient can be metabolized for energy without the direct
    involvement of oxygen?
  sentences:
  - Blood lactate concentrations reflect lactate production and clearance. The clearance
    of lactate from the blood reflects a return to homeostasis and thus a person’s
    ability to recover.
  - 'Of the three main macronutrients—carbohydrates, proteins, and fats—only carbohydrates
    can be metabolized for energy without the direct involvement of oxygen. Carbohydrates
    are critical during anaerobic metabolism. Energy stored in the chemical bonds
    of adenosine triphosphate (ATP) is used to power muscular activity. The replenishment
    of ATP in human skeletal muscle is accomplished by three basic energy systems:
    the phosphagen, glycolytic, and oxidative systems.'
  - Bioenergetics, or the flow of energy in a biological system, concerns primarily
    the conversion of macronutrients—carbohydrates, proteins, and fats, which contain
    chemical energy—into biologically usable forms of energy, defined as the ability
    or capacity to perform work. It is the breakdown of the chemical bonds in these
    macronutrients that provides the energy necessary to perform biological work.
- source_sentence: what kind of reactions release energy and are usually catabolic?
  sentences:
  - It is the breakdown of the chemical bonds in these macronutrients that provides
    the energy necessary to perform biological work. The breakdown of large molecules
    into smaller molecules, associated with the release of energy, is termed catabolism.
    The synthesis of larger molecules from smaller molecules can be accomplished using
    the energy released from catabolic reactions; this building-up process is termed
    anabolism. The breakdown of proteins into amino acids is an example of catabolism,
    while the formation of proteins from amino acids is an anabolic process. Exergonic
    reactions are energy-releasing reactions and are generally catabolic. Endergonic
    reactions require energy and include anabolic processes and the contraction of
    muscle.
  - Exergonic reactions are energy-releasing reactions and are generally catabolic.
  - Metabolism is the total of all the catabolic or exergonic and anabolic or endergonic
    reactions in a biological system. Energy derived from catabolic or exergonic reactions
    is used to drive anabolic or endergonic reactions through an intermediate molecule,
    adenosine triphosphate (ATP). Adenosine triphosphate allows the transfer of energy
    from exergonic to endergonic reactions.
- source_sentence: What metabolites allosterically inhibit and activate pyruvate kinase?
  sentences:
  - Three important glycolytic enzymes are hexokinase, phosphofructokinase, and pyruvate
    kinase. All three of these enzymes are regulatory enzymes in glycolysis because
    each has important allosteric binding sites. Allosteric regulation occurs when
    the end product of a reaction or series of reactions feeds back to regulate the
    turnover rate of key enzymes in the metabolic pathways. Allosteric inhibition
    occurs when an end product binds to the regulatory enzyme and decreases its turnover
    rate and slows product formation. Allosteric activation occurs when an 'activator'
    binds with the enzyme and increases its turnover rate.
  - 'Three basic energy systems exist in mammalian muscle cells to replenish ATP:
    the phosphagen system, glycolysis, and the oxidative system. Anaerobic processes
    do not require the presence of oxygen, whereas aerobic mechanisms depend on oxygen.
    The phosphagen system and the first phase of glycolysis are anaerobic mechanisms
    that occur in the sarcoplasm of a muscle cell. The Krebs cycle, electron transport,
    and the rest of the oxidative system are aerobic mechanisms that occur in the
    mitochondria of muscle cells and require oxygen as the terminal electron receptor.'
  - Pyruvate kinase catalyzes the conversion of phosphoenolpyruvate to pyruvate and
    is the final regulatory enzyme. Pyruvate kinase is allosterically inhibited by
    ATP and acetyl-CoA and activated by high concentrations of AMP and fructose-1,6-bisphosphate.
- source_sentence: how does oxygen and carbon dioxide move between the lungs and blood?
  sentences:
  - The way in which carbon dioxide is removed from the system has some similarities
    to oxygen transport, but the vast amount of carbon dioxide is removed by a more
    complex process. After carbon dioxide is formed in the cell, it is transported
    out of the cell by diffusion and subsequently transported to the lungs. Only a
    limited quantity of carbon dioxide—about 5% of that produced during metabolism—is
    carried in the plasma; similar to the situation with oxygen, this limited amount
    of carbon dioxide contributes to establishing the partial pressure of carbon dioxide
    in blood. Some carbon dioxide is also transported via hemoglobin, but this too
    is a limited amount.
  - The cardiovascular system transports nutrients and removes waste products while
    helping to maintain the environment for all the body’s functions. The blood transports
    oxygen from the lungs to the tissues for use in cellular metabolism, and it transports
    carbon dioxide—the most abundant by-product of metabolism—from the tissues to
    the lungs, where it is removed from the body. Two paramount functions of blood
    are the transport of oxygen from the lungs to the tissues for use in cellular
    metabolism and the removal of carbon dioxide, the most abundant by-product of
    metabolism, from the tissues to the lungs. The transport of oxygen is accomplished
    by hemoglobin, the iron-protein molecule carried by the red blood cells. Hemoglobin
    also has an additional important role as an acid–base buffer, a regulator of hydrogen
    ion concentration is crucial to the rates of chemical reactions in cells. Red
    blood cells are the major component of blood. Red blood cells contain a large
    quantity of carbonic anhydrase. Carbonic anhydrase catalyzes the reaction between
    carbon dioxide and water to facilitate carbon dioxide removal.
  - The discharge of an action potential from a motor nerve signals the release of
    calcium from the sarcoplasmic reticulum into the myofibril, causing tension development
    in muscle. In its simplest form, the sliding-filament theory states that the actin
    filaments at each end of the sarcomere slide inward on myosin filaments, pulling
    the Z-lines toward the center of the sarcomere and thus shortening the muscle
    fiber. As actin filaments slide over myosin filaments, both the H-zone and I-band
    shrink. The flexion of myosin cross-bridges pulling on the actin filaments is
    responsible for the movement of the actin filament.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
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
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
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
    'how does oxygen and carbon dioxide move between the lungs and blood?',
    'The cardiovascular system transports nutrients and removes waste products while helping to maintain the environment for all the body’s functions. The blood transports oxygen from the lungs to the tissues for use in cellular metabolism, and it transports carbon dioxide—the most abundant by-product of metabolism—from the tissues to the lungs, where it is removed from the body. Two paramount functions of blood are the transport of oxygen from the lungs to the tissues for use in cellular metabolism and the removal of carbon dioxide, the most abundant by-product of metabolism, from the tissues to the lungs. The transport of oxygen is accomplished by hemoglobin, the iron-protein molecule carried by the red blood cells. Hemoglobin also has an additional important role as an acid–base buffer, a regulator of hydrogen ion concentration is crucial to the rates of chemical reactions in cells. Red blood cells are the major component of blood. Red blood cells contain a large quantity of carbonic anhydrase. Carbonic anhydrase catalyzes the reaction between carbon dioxide and water to facilitate carbon dioxide removal.',
    'The way in which carbon dioxide is removed from the system has some similarities to oxygen transport, but the vast amount of carbon dioxide is removed by a more complex process. After carbon dioxide is formed in the cell, it is transported out of the cell by diffusion and subsequently transported to the lungs. Only a limited quantity of carbon dioxide—about 5% of that produced during metabolism—is carried in the plasma; similar to the situation with oxygen, this limited amount of carbon dioxide contributes to establishing the partial pressure of carbon dioxide in blood. Some carbon dioxide is also transported via hemoglobin, but this too is a limited amount.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6713, 0.5596],
#         [0.6713, 1.0000, 0.6922],
#         [0.5596, 0.6922, 1.0000]])
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

* Size: 26,300 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                             | negative                                                                             |
  |:--------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                               | string                                                                               |
  | details | <ul><li>min: 10 tokens</li><li>mean: 21.43 tokens</li><li>max: 44 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 124.98 tokens</li><li>max: 342 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 118.91 tokens</li><li>max: 350 tokens</li></ul> |
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

- `per_device_train_batch_size`: 16
- `learning_rate`: 2e-05
- `warmup_steps`: 0.1
- `gradient_checkpointing`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
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
| 0.0998 | 164  | 0.5555        |
| 0.1995 | 328  | 0.4441        |
| 0.2993 | 492  | 0.4033        |
| 0.3990 | 656  | 0.3448        |
| 0.4988 | 820  | 0.2928        |
| 0.5985 | 984  | 0.2743        |
| 0.6983 | 1148 | 0.2441        |
| 0.7981 | 1312 | 0.2120        |
| 0.8978 | 1476 | 0.2167        |
| 0.9976 | 1640 | 0.1907        |
| 1.0973 | 1804 | 0.1354        |
| 1.1971 | 1968 | 0.1364        |
| 1.2968 | 2132 | 0.1281        |
| 1.3966 | 2296 | 0.1330        |
| 1.4964 | 2460 | 0.1280        |
| 1.5961 | 2624 | 0.1166        |
| 1.6959 | 2788 | 0.1139        |
| 1.7956 | 2952 | 0.1015        |
| 1.8954 | 3116 | 0.1062        |
| 1.9951 | 3280 | 0.1058        |
| 2.0949 | 3444 | 0.0867        |
| 2.1946 | 3608 | 0.0726        |
| 2.2944 | 3772 | 0.0814        |
| 2.3942 | 3936 | 0.0813        |
| 2.4939 | 4100 | 0.0805        |
| 2.5937 | 4264 | 0.0704        |
| 2.6934 | 4428 | 0.0712        |
| 2.7932 | 4592 | 0.0769        |
| 2.8929 | 4756 | 0.0671        |
| 2.9927 | 4920 | 0.0862        |


### Training Time
- **Training**: 6.8 hours

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