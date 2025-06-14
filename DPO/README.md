This folder contrains the DPO code for tuning LLM on stackoverflow data. Original example link here : ([Original code](https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts)). Updated the code for proper logging and made it configurable.

## 1) SFT
#### 1.1) Model Details
Model name : Qwen2.0_0.5B
#### 1.2) Training Insights
##### 1.2.1) Training logs
{'loss': 2.2277, 'grad_norm': 0.3126915991306305, 'learning_rate': 1.501e-05, 'num_tokens': 4096000.0, 'mean_token_accuracy': 0.5384662748575211, 'epoch': 0.0}
{'loss': 2.1707, 'grad_norm': 0.34503886103630066, 'learning_rate': 1.0009999999999999e-05, 'num_tokens': 8192000.0, 'mean_token_accuracy': 0.550085776746273, 'epoch': 0.0}
{'loss': 2.1707, 'grad_norm': 0.3259875476360321, 'learning_rate': 5.01e-06, 'num_tokens': 12288000.0, 'mean_token_accuracy': 0.5487971640229226, 'epoch': 0.0}
{'loss': 2.164, 'grad_norm': 0.3930286765098572, 'learning_rate': 1e-08, 'num_tokens': 16384000.0, 'mean_token_accuracy': 0.548846530020237, 'epoch': 0.0}
{'train_runtime': 4517.0833, 'train_samples_per_second': 3.542, 'train_steps_per_second': 0.443, 'train_loss': 2.1832706298828124, 'epoch': 0.0}

##### 1.2.2) Total time taken
1 hour, 15 minutes, 23.54 seconds
