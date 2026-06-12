---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:26300
- loss:MultipleNegativesRankingLoss
base_model: TaylorAI/bge-micro-v2
widget:
- source_sentence: What are the anabolic effects of high-intensity resistance training
    on muscle mass in older men?
  sentences:
  - Analysis of the MHCs showed that in the early phase of training, IIx MHCs were
    replaced with IIa MHCs. Detraining results in an increase in Type IIx fibers and
    a reduction in Type IIa fibers. Detraining may result in an overshoot of Type
    IIx fibers (i.e., higher IIx percentages than observed pretraining). Transformation
    from Type I to Type II or vice versa appears less probable.
  - Regular participation in a resistance training program also seems to have profound
    anabolic effects in older populations. Computed tomography and muscle biopsy analysis
    showed evidence of muscle hypertrophy in older men who participated in a high-intensity
    resistance training program. Other investigations involving older adults have
    shown that resistance training can improve nitrogen retention, which can have
    a positive effect on muscle protein metabolism.
  - Resistance training has also been shown to have an important effect on energy
    balance in older adults, as evidenced by an increase in the resting metabolic
    rate of men and women who resistance train. It is noteworthy that dietary modifications
    (a change in total food intake or selected nutrients) in older men who resistance
    trained had a positive effect on muscle hypertrophy.
- source_sentence: how quickly does strength decline when you stop training, and what
    are the main factors involved?
  sentences:
  - Factors that may contribute to the age-related decline in muscle strength and
    power include reductions in muscle mass, nervous system changes, hormonal changes,
    poor nutrition, and physical inactivity.
  - The intensity level of the Lateral Push-off is low. The direction of the jump
    in the Lateral Push-off is vertical. Equipment for the Lateral Push-off includes
    a plyometric box, 6 to 18 inches (15-46 cm) high.
  - Detraining is the cessation of anaerobic training or a substantial reduction in
    frequency, volume, intensity, or any combination of those three variables that
    results in decrements in performance and loss of some of the physiological adaptations
    associated with resistance training. Detraining may occur in as few as two weeks
    and possibly sooner in well-trained individuals. In recreationally trained men,
    very little change is seen during the first six weeks of detraining. Strength
    reductions appear related to neural mechanisms initially, with atrophy predominating
    as the detraining period extends. The amount of muscle strength retained is rarely
    lower than pre-training values, indicating that resistance training has a residual
    effect when the stimulus is removed. When the athlete returns to training, the
    rate of strength reattainment is high, suggesting the concept of ‘muscle memory’.”
- source_sentence: What are the primary purposes of athletic testing?
  sentences:
  - During the needs analysis (step 1, p. 382), the strength and conditioning professional
    is challenged to choose the primary goal of the resistance training program based
    on the athlete’s testing results, the movement and physiological analysis of the
    sport, and the priorities of the athlete’s sport season. Once decided on, the
    training goal can be applied to determine specific load and repetition assignments
    via the RM continuum, a percentage of the 1RM (either directly tested or estimated),
    or the results of multiple-RM testing.
  - McNaughton and colleagues (138, 139) and Coombes and McNaughton (49) have demonstrated
    improved total work capacity, peak power, peak torque, and strength from acute
    SB supplementation in men and women. Edge and colleagues (62) examined the effects
    of eight weeks of SB ingestion during high-intensity interval training on cycling
    performance parameters. The results of Edge and colleagues (62) exhibited a significant
    increase in the effects of high-intensity interval training on anaerobic threshold,
    time to fatigue, and peak power with SB versus placebo. Edge and colleagues (62)
    suggested that increasing the ability to regulate H+, thus maintaining intracellular
    pH, may be vital to enhancing the benefits of intense training.
  - Testing can be used to assess athletic talent, identify physical abilities and
    areas in need of improvement, set goals, and evaluate progress.
- source_sentence: What are the anticipated effects of recombinant insulin-like growth
    factor I (IGF-I) on the human body?
  sentences:
  - Insulin-like growth factor I is also produced in the muscles themselves in response
    to overload and stretch. IGF-I produced in muscle is called mechano growth factor
    (MGF) and exerts autocrine functions (52, 117). It has been suggested that autocrine
    actions of MGF are the primary actions of IGF-I in muscle.
  - Insulin-like growth factor I is now being synthesized using recombinant DNA technology
    and will likely produce the same effects as HGH.
  - 'The equation for calculating estimated body fat percentage (%BF) from body density
    (Db) for Japanese native males aged 18-48 years is: %BF = (4.97/Db) – 4.52. The
    equation for calculating estimated body fat percentage (%BF) from body density
    (Db) for Japanese native females aged 18-48 years is: %BF = (4.76/Db) – 4.28.
    The equation for calculating estimated body fat percentage (%BF) from body density
    (Db) for Japanese native males aged 61-78 years is: %BF = (4.87/Db) – 4.41. The
    equation for calculating estimated body fat percentage (%BF) from body density
    (Db) for Japanese native females aged 61-78 years is: %BF = (4.95/Db) – 4.50.'
- source_sentence: why do some weightlifting formulas try to account for body weight
    when comparing performance?
  sentences:
  - Body mass index (BMI) is the preferred body composition assessment for obese individuals.
    Skinfold assessment becomes inaccurate because of the size of the skinfold and
    the lack of standardized formulas for obese adults.
  - At the end of the set, step toward the rack and place the bar in the supports.
  - Various formulas have been derived to compare loads lifted more equitably. One
    would expect the weight category with the largest number of competitors to produce
    the best performers. Because of the bell-shaped curve describing the normal distribution
    of anthropometric characteristics among the population, the body weights of a
    majority of people are clustered close to the mean. The classic formula’s determination
    that the performances of medium-weight athletes are usually the best may indeed
    be unbiased. Other formulas have been developed because the classic formula seemed
    to favor athletes of middle body weight over lighter and heavier athletes.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on TaylorAI/bge-micro-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [TaylorAI/bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [TaylorAI/bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2) <!-- at revision 3edf6d7de0faa426b09780416fe61009f26ae589 -->
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
    'why do some weightlifting formulas try to account for body weight when comparing performance?',
    'Various formulas have been derived to compare loads lifted more equitably. One would expect the weight category with the largest number of competitors to produce the best performers. Because of the bell-shaped curve describing the normal distribution of anthropometric characteristics among the population, the body weights of a majority of people are clustered close to the mean. The classic formula’s determination that the performances of medium-weight athletes are usually the best may indeed be unbiased. Other formulas have been developed because the classic formula seemed to favor athletes of middle body weight over lighter and heavier athletes.',
    'Body mass index (BMI) is the preferred body composition assessment for obese individuals. Skinfold assessment becomes inaccurate because of the size of the skinfold and the lack of standardized formulas for obese adults.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.4998, 0.3230],
#         [0.4998, 1.0000, 0.3806],
#         [0.3230, 0.3806, 1.0000]])
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

### Evaluation Dataset

#### Unnamed Dataset

* Size: 1,219 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                            | negative                                                                            |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              | string                                                                              |
  | details | <ul><li>min: 8 tokens</li><li>mean: 21.48 tokens</li><li>max: 48 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 92.63 tokens</li><li>max: 333 tokens</li></ul> | <ul><li>min: 8 tokens</li><li>mean: 109.13 tokens</li><li>max: 478 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                          | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:--------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What is the primary function of anabolic hormones in relation to the body's response to heavy resistance training?</code> | <code>Strength and conditioning professionals and athletes appreciate the importance of anabolic hormones for mediating changes in the body and helping with the adaptive response to heavy resistance training.<br>The goal of this chapter is to provide an initial glimpse into the endocrine system that mediates the changes in the body with training.</code>                                                                                                                                                                                                                                                       | <code>It is important for strength and conditioning professionals to have a basic understanding of the hormonal responses to resistance exercise. Knowledge of hormonal responses to resistance exercise increases insight into how an exercise prescription can enable hormones to mediate optimal adaptations to resistance training. Although resistance training is the only natural stimulus that causes increases in lean tissue mass, dramatic differences exist among resistance training programs in their ability to produce increases in muscle and connective tissue size. The type of resistance training workout used dictates the hormonal responses. Tissue adaptations are influenced by the changes in circulating hormonal concentrations following exercise. Understanding this natural anabolic activity that takes place in the athlete’s body is fundamental to successful recovery, adaptation, program design, training progression, and ultimately athletic performance. It has been theorized that the endocrine sy...</code> |
  | <code>what's the endocrine system's role in how our bodies change when we lift weights?</code>                                  | <code>Strength and conditioning professionals and athletes appreciate the importance of anabolic hormones for mediating changes in the body and helping with the adaptive response to heavy resistance training.<br>The goal of this chapter is to provide an initial glimpse into the endocrine system that mediates the changes in the body with training.</code>                                                                                                                                                                                                                                                       | <code>The endocrine system supports the normal homeostatic function of the human body and helps it respond to external stimuli. Hans Selye, a Canadian endocrinologist, unknowingly provided the theoretical basis for periodization with his work on the adrenal gland and the role of stress hormones in the adaptation to stress, distress, and illness. Hans Selye coined the term General Adaptation Syndrome to describe how the adrenal gland responds with an initial alarm reaction followed by a reduction of an organism’s function in response to a noxious stimulus. The key to continued adaptation to the stress is the timely removal of the stimulus so that the organism’s function can recover.</code>                                                                                                                                                                                                                                                                                                                                |
  | <code>What physiological systems support neuromuscular activity during physical exercise?</code>                                | <code>Physical exercise and sport performance involve effective, purposeful movements of the body. These movements result from the forces developed in muscles, which, acting through lever systems of the skeleton, move the various body parts. These skeletal muscles are under the control of the cerebral cortex, which, working through motor neurons, activates the skeletal muscle cells or fibers. Support for this neuromuscular activity involves continuous delivery of oxygen to and removal of carbon dioxide from working tissues through activities of the respiratory and cardiovascular systems.</code> | <code>Understanding how the individual systems of the human body adapt to physical activity provides a knowledge base from which the strength and conditioning professional can predict the outcome of a specific training program. Anaerobic training may elicit adaptations along the neuromuscular chain beginning in the higher brain centers and continuing down to the level of individual muscle fibers (figure 5.1). An increase in neural drive is critical to the athlete striving to maximize strength and power. The increase in neural drive is thought to occur via increases in agonist (i.e., those major muscles involved in a specific movement or exercise) muscle recruitment, firing rate, and the timing and pattern of discharge during high-intensity muscular contractions. A reduction in inhibitory mechanisms (i.e., from Golgi tendon organs) is also thought to occur. Although it is not clear how these mechanisms coexist, it is apparent that neural adaptations are complex and typically occur before stru...</code> |
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
- `per_device_eval_batch_size`: 16
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
- `per_device_eval_batch_size`: 16
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
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0998 | 164  | 0.9015        | -               |
| 0.1995 | 328  | 0.6218        | -               |
| 0.2993 | 492  | 0.5380        | -               |
| 0.3990 | 656  | 0.5147        | -               |
| 0.4988 | 820  | 0.4409        | -               |
| 0.5985 | 984  | 0.4103        | -               |
| 0.6983 | 1148 | 0.4035        | -               |
| 0.7981 | 1312 | 0.3600        | -               |
| 0.8978 | 1476 | 0.3601        | -               |
| 0.9976 | 1640 | 0.3187        | -               |
| 1.0    | 1644 | -             | 0.3094          |
| 1.0973 | 1804 | 0.2730        | -               |
| 1.1971 | 1968 | 0.2860        | -               |
| 1.2968 | 2132 | 0.2602        | -               |
| 1.3966 | 2296 | 0.2628        | -               |
| 1.4964 | 2460 | 0.2553        | -               |
| 1.5961 | 2624 | 0.2571        | -               |
| 1.6959 | 2788 | 0.2355        | -               |
| 1.7956 | 2952 | 0.2341        | -               |
| 1.8954 | 3116 | 0.2186        | -               |
| 1.9951 | 3280 | 0.2267        | -               |
| 2.0    | 3288 | -             | 0.3089          |
| 2.0949 | 3444 | 0.2077        | -               |
| 2.1946 | 3608 | 0.1926        | -               |
| 2.2944 | 3772 | 0.1960        | -               |
| 2.3942 | 3936 | 0.1947        | -               |
| 2.4939 | 4100 | 0.2007        | -               |
| 2.5937 | 4264 | 0.1872        | -               |
| 2.6934 | 4428 | 0.1865        | -               |
| 2.7932 | 4592 | 0.2055        | -               |
| 2.8929 | 4756 | 0.1749        | -               |
| 2.9927 | 4920 | 0.1919        | -               |
| 3.0    | 4932 | -             | 0.3122          |


### Training Time
- **Training**: 1.4 hours
- **Evaluation**: 1.2 minutes
- **Total**: 1.4 hours

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