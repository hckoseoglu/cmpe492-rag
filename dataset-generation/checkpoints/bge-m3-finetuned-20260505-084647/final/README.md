---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:28010
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-m3
widget:
- source_sentence: what causes muscle fatigue during exercise, and what's the role
    of lactate in that?
  sentences:
  - The formation of lactate from pyruvate is catalyzed by the enzyme lactate dehydrogenase.
    At physiological pH (i.e., near 7), the lactic acid molecule cannot exist. Lactate,
    not lactic acid, is the result of the lactate dehydrogenase reaction. Although
    the muscular fatigue experienced during exercise often correlates with high tissue
    concentrations of lactate, lactate is not the cause of fatigue. The H+ accumulation
    as a result of lactate formation reduces the intracellular pH, inhibits glycolytic
    reactions, and directly interferes with muscle’s excitation-contraction coupling.
  - The oxidative metabolism of blood glucose and muscle glycogen begins with glycolysis.
    If oxygen is present in sufficient quantities, the end product of glycolysis,
    pyruvate, is not converted to lactate but is transported to the mitochondria.
    In the mitochondria, pyruvate is taken up and enters the Krebs cycle, citric acid
    cycle, or tricarboxylic acid cycle. The Krebs cycle is a series of reactions that
    continues the oxidation of the substrate begun in glycolysis and produces two
    ATP indirectly from guanine triphosphate (GTP) via substrate-level phosphorylation
    for each molecule of glucose.
  - The H+ accumulation as a result of lactate formation reduces the intracellular
    pH, inhibits glycolytic reactions, and directly interferes with muscle’s excitation-contraction
    coupling. The decrease in pH inhibits the enzymatic turnover rate of the cell’s
    energy systems. The process of an exercise-induced decrease in pH is referred
    to as metabolic acidosis and may be responsible for much of the peripheral fatigue
    that occurs during exercise.
- source_sentence: how do the electrical signals travel from the AV node to the ventricles
    and what's the purpose of this pathway?
  sentences:
  - The AV node is located in the posterior septal wall of the right atrium. The left
    and right bundle branches lead from the AV bundle into the ventricles. Except
    for their initial portion, where they penetrate the AV barrier, these conduction
    fibers have functional characteristics quite opposite those of the AV nodal fibers.
    They are large and transmit impulses at a much higher velocity than the AV nodal
    fibers. Because these fibers give way to the Purkinje fibers, which more completely
    penetrate the ventricles, the impulse travels quickly throughout the entire ventricular
    system and causes both ventricles to contract at approximately the same time.
  - The SA node is a small area of specialized muscle tissue located in the upper
    lateral wall of the right atrium. The fibers of the SA node are continuous with
    the muscle fibers of the atrium. Each electrical impulse that begins in the SA
    node normally spreads immediately into the atria. The conductive system is organized
    so that the impulse does not travel into the ventricles too rapidly, allowing
    time for the atria to contract and empty blood into the ventricles before ventricular
    contraction begins. The AV node and its associated conductive fibers primarily
    delay each impulse entering into the ventricles.
  - 'The phosphagen system provides ATP primarily for short-term, high-intensity activities
    (e.g., resistance training and sprinting) and is active at the start of all exercise
    regardless of intensity. This energy system relies on the hydrolysis of ATP (Equation
    2.1) and breakdown of another high-energy phosphate molecule called creatine phosphate
    (CP), also called phosphocreatine (PCr). Creatine kinase is the enzyme that catalyzes
    the synthesis of ATP from CP and ADP in the following reaction: ADP + CP  ATP
    + Creatine (2.2). Creatine phosphate supplies a phosphate group that combines
    with ADP to replenish ATP. The creatine kinase reaction provides energy at a high
    rate; however, because CP is stored in relatively small amounts, the phosphagen
    system cannot be the primary supplier of energy for continuous, long-duration
    activities.'
- source_sentence: What are the primary energy sources and the limiting factor of
    the phosphagen system?
  sentences:
  - The phosphagen energy system primarily supplies ATP for high-intensity activities
    of short duration (e.g., 100 m dash), the glycolytic system for moderate- to high-intensity
    activities of short to medium duration (e.g., 400 m dash), and the oxidative system
    for low-intensity activities of long duration (e.g., marathon).
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
  - 'Three basic energy systems exist in mammalian muscle cells to replenish ATP:
    the phosphagen system, glycolysis, and the oxidative system. Anaerobic processes
    do not require the presence of oxygen, whereas aerobic mechanisms depend on oxygen.
    The phosphagen system and the first phase of glycolysis are anaerobic mechanisms
    that occur in the sarcoplasm of a muscle cell. The Krebs cycle, electron transport,
    and the rest of the oxidative system are aerobic mechanisms that occur in the
    mitochondria of muscle cells and require oxygen as the terminal electron receptor.'
- source_sentence: how does the concentration of reactants like ADP affect the rate
    of ATP replenishment during exercise?
  sentences:
  - Diffusion results from the movement of gas from high concentration to low concentration.
    At the tissue level, where oxygen is utilized in metabolism and carbon dioxide
    is produced, the partial pressures of these gases in some instances differ considerably
    from those in arterial blood. At rest, the partial pressure of oxygen in the fluid
    immediately outside a muscle cell rapidly drops from 100 mmHg in arterial blood
    to as low as 40 mmHg, while the partial pressure of carbon dioxide is elevated
    above that of arterial blood to about 46 mmHg.
  - 'The phosphagen system provides ATP primarily for short-term, high-intensity activities
    (e.g., resistance training and sprinting) and is active at the start of all exercise
    regardless of intensity. This energy system relies on the hydrolysis of ATP (Equation
    2.1) and breakdown of another high-energy phosphate molecule called creatine phosphate
    (CP), also called phosphocreatine (PCr). Creatine kinase is the enzyme that catalyzes
    the synthesis of ATP from CP and ADP in the following reaction: ADP + CP  ATP
    + Creatine (2.2). Creatine phosphate supplies a phosphate group that combines
    with ADP to replenish ATP. The creatine kinase reaction provides energy at a high
    rate; however, because CP is stored in relatively small amounts, the phosphagen
    system cannot be the primary supplier of energy for continuous, long-duration
    activities.'
  - 'In addition, Type II (fast-twitch) muscle fibers contain higher concentrations
    of CP than Type I (slow-twitch) fibers; thus, individuals with higher percentages
    of Type II fibers may be able to replenish ATP faster through the phosphagen system
    during anaerobic, explosive exercise. Another important single-enzyme reaction
    that can rapidly replenish ATP is the adenylate kinase (also called myokinase)
    reaction: 2ADP  ATP + AMP (2.3). This reaction is particularly important because
    AMP, a product of the adenylate kinase (myokinase) reaction, is a powerful stimulant
    of glycolysis.'
- source_sentence: what's the normal heart rate of the SA node, and how does it compare
    to the AV node and ventricular fibers?
  sentences:
  - The myosin filaments (thick filaments about 16 nm in diameter, about 1/10,000
    the diameter of a hair) contain up to 200 myosin molecules. Globular heads called
    cross-bridges protrude away from the myosin filament at regular intervals. The
    actin filaments (thin filaments about 6 nm in diameter) consist of two strands
    arranged in a double helix. Myosin and actin filaments are organized longitudinally
    in the smallest contractile unit of skeletal muscle, the sarcomere.
  - The inherent rhythmicity and conduction properties of the myocardium are influenced
    by the cardiovascular center of the medulla, which transmits signals to the heart
    through the sympathetic and parasympathetic nervous systems. The atria are supplied
    with a large number of both sympathetic and parasympathetic neurons, whereas the
    ventricles receive sympathetic fibers almost exclusively. Stimulation of the sympathetic
    nerves accelerates depolarization of the SA node (the chronotropic effect), which
    causes the heart to beat faster. Stimulation of the parasympathetic nervous system
    slows the rate of SA node discharge, which slows the heart rate.
  - The normal discharge rate of the sinoatrial (SA) node ranges from 60 to 80 times
    per minute. Aerobic endurance training results in a significantly slower discharge
    rate of the SA node due to an increase in parasympathetic tone. Increased stroke
    volume also affects the resting heart rate—more blood is pumped per contraction
    so that the heart contracts less frequently.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-m3

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) <!-- at revision 5617a9f61b028005a4858fdac845db406aefb181 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 1024 dimensions
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
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'embedding_dimension': 1024, 'pooling_mode': 'cls', 'include_prompt': True})
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
    "what's the normal heart rate of the SA node, and how does it compare to the AV node and ventricular fibers?",
    'The normal discharge rate of the sinoatrial (SA) node ranges from 60 to 80 times per minute. Aerobic endurance training results in a significantly slower discharge rate of the SA node due to an increase in parasympathetic tone. Increased stroke volume also affects the resting heart rate—more blood is pumped per contraction so that the heart contracts less frequently.',
    'The inherent rhythmicity and conduction properties of the myocardium are influenced by the cardiovascular center of the medulla, which transmits signals to the heart through the sympathetic and parasympathetic nervous systems. The atria are supplied with a large number of both sympathetic and parasympathetic neurons, whereas the ventricles receive sympathetic fibers almost exclusively. Stimulation of the sympathetic nerves accelerates depolarization of the SA node (the chronotropic effect), which causes the heart to beat faster. Stimulation of the parasympathetic nervous system slows the rate of SA node discharge, which slows the heart rate.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6534, 0.3048],
#         [0.6534, 1.0000, 0.2724],
#         [0.3048, 0.2724, 1.0000]])
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

* Size: 28,010 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                             | negative                                                                             |
  |:--------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                               | string                                                                               |
  | details | <ul><li>min: 11 tokens</li><li>mean: 23.11 tokens</li><li>max: 51 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 138.18 tokens</li><li>max: 385 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 132.08 tokens</li><li>max: 407 tokens</li></ul> |
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
- `gradient_accumulation_steps`: 2
- `fp16`: True
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
- `gradient_accumulation_steps`: 2
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
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
| 0.1999 | 175  | 0.4227        |
| 0.3998 | 350  | 0.2809        |
| 0.5997 | 525  | 0.2262        |
| 0.7995 | 700  | 0.1883        |
| 0.9994 | 875  | 0.1502        |
| 1.1987 | 1050 | 0.0903        |
| 1.3986 | 1225 | 0.0804        |
| 1.5985 | 1400 | 0.0789        |
| 1.7984 | 1575 | 0.0700        |
| 1.9983 | 1750 | 0.0676        |
| 2.1976 | 1925 | 0.0401        |
| 2.3975 | 2100 | 0.0402        |
| 2.5974 | 2275 | 0.0351        |
| 2.7973 | 2450 | 0.0392        |
| 2.9971 | 2625 | 0.0364        |


### Training Time
- **Training**: 1.5 hours

### Framework Versions
- Python: 3.10.20
- Sentence Transformers: 5.4.1
- Transformers: 5.7.0
- PyTorch: 2.10.0+cu128
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