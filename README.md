# Complex Locomotion Skill Learning via Differentiable Physics
[[OpenReview](https://openreview.net/forum?id=YpBHDlalKDG)] [[arXiv](https://arxiv.org/abs/2206.02341)]

## RL

### Train
`python3 main_rl.py --config_file cfg/sim_config_RL_robot2_vh.json --env-name "RL_Multitask" --algo ppo --use-gae --use-linear-lr-decay --train`

### Evaluate 
`python3 main_rl.py --env-name "RL_Multitask" --algo ppo --use-gae --use-linear-lr-decay --evaluate --evaluate_path "saved_results/sim_config_RL_robot2_vh/DiffTaichi_RL/" `

## DiffPhy

### Train
`python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vh.json --train`

### Evaluate
`python3 main_diff_phy.py --config_file cfg/sim_config_DiffPhy_robot2_vh.json --evaluate --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vh/DiffTaichi_DiffPhy`

###  Interactive
`python3 interactive.py --config_file cfg/sim_config_DiffPhy_robot2_vh.json --memory 1.0`


## Output plots

### DiffPhy Only

#### Multi-Plots
`python3 scripts/export_tensorboard_data.py --our_file_path saved_results/sim_config_DiffPhy_robot4_vh/DiffTaichi_DiffPhy/ --no-rl --error-bar --task tvh`

#### Single-Plot
`python3 scripts/export_tensorboard_data.py --our_file_path saved_results/sim_config_DiffPhy_robot4_vh/DiffTaichi_DiffPhy/ --no-rl --error-bar --task tvh --draw-single`

### DiffPhy and RL

#### Multi-Plots
`python3 scripts/export_tensorboard_data.py --our_file_path saved_results/sim_config_DiffPhy_robot4_vh/DiffTaichi_DiffPhy/ --rl_file_path saved_results/sim_config_RL_robot4_vh/DiffTaichi_RL/ --task tvh`

#### Single-Plot
`python3 scripts/export_tensorboard_data.py --our_file_path saved_results/sim_config_DiffPhy_robot4_vh/DiffTaichi_DiffPhy/ --rl_file_path saved_results/sim_config_RL_robot4_vh/DiffTaichi_RL/ --task tvh --draw-single --error-bar`


## 3D
python3 main_diff_phy.py --config_file cfg3d/sim_config_DiffPhy_robot2_vh.json --train
python3 main_diff_phy.py --config_file cfg3d/sim_config_DiffPhy_robot2_vh.json --evaluate --no-tensorboard-train --evaluate_path saved_results/sim_config_DiffPhy_robot2_vh/DiffTaichi_DiffPhy


## Generate scripts
#### Adam grid search
python3 scripts/json_gen_multiple_robots.py --robots 2 --output-name run_scripts_adam_grid_search_b1_b2_0_9_0_999.sh --adam-grid-search --adam-b1 0.82 0.68 0.43 --adam-b2 0.9 0.999 --gpu-id 0

## Cite this work
```
@article{liu2021complex,
  title={Complex Locomotion Skill Learning via Differentiable Physics},
  author={Liu, Jiancheng and Fang, Yu and Zhang, Mingrui and Zhang, Jiasheng and Ma, Yidong and Li, Minchen and Hu, Yuanming and Jiang, Chenfanfu and Liu, Tiantian},
  year={2021}
}
```
