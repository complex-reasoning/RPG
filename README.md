## RPG: On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)

- **RPG (Regularized policy gradient)** is a systematic framework for deriving and analyzing KL-regularized policy gradient methods in the online reinforcement learning (RL) setting. We derive policy gradients and corresponding surrogate loss functions for objectives regularized by both forward and reverse KL divergences, considering both normalized and unnormalized policy distributions. Furthermore, we present derivations for fully differentiable loss functions as well as REINFORCE-style gradient estimators, accommodating diverse algorithmic needs. This repository provides tool for data preparation and RLHF process for the paper "[On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](https://arxiv.org/abs/2505.17508)".
- Authors: [Yifan Zhang](https://yifzhang.com)\*, [Yifeng Liu](https://lauyikfung.github.io)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ), [Yang Yuan](https://scholar.google.com/citations?user=7o4wtKEAAAAJ&hl=en), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/), [Andrew Chi-Chih Yao](https://en.wikipedia.org/wiki/Andrew_Yao)

[[Webpage](https://complex-reasoning.github.io/RPG)] [[Huggingface](https://huggingface.co/papers/2505.17508)]

## Table of Contents


  - [On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](#on-the-design-of-kl-regularized-policy-gradient-algorithms-for-llm-reasoning)
    - [Table of Contents](#table-of-contents)
    - [Features](#features)
    - [Hardware Requirements](#hardware-requirements)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
    - [Repeat the Experiments](#repeat-the-experiments)
    - [Acknowledgements](#acknowledgements)
    - [Star History](#star-history)
    - [Citation](#citation)

  ## Features

  - **Comprehensiveness**: We derive policy gradients and corresponding surrogate loss functions for objectives regularized by Forward and Reverse KL divergences, considering both standard normalized (KL) and unnormalized (UKL) forms.
  - **Flexible RL setting**: We systematically provide derivations for fully differentiable loss functions (offering connections to variational inference) and REINFORCE-style gradient estimators (employing the stop-gradient operator). These are developed for the online setting, using off-policy gradient estimation via importance sampling from a prior policy $\pi_{\mathrm{old}}$.

## Hardware Requirements

A100 and H100 are recommended. At least 8*80G VRAM is needed.

## Installation

Ensure you have Python 3.10 or higher installed. It's recommended to use a virtual environment to manage dependencies.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/complex-reasoning/RPG.git
   cd RPG
   ```

2. **Follow the instruction of verl**

   Follow the documentation in [Welcome to verl’s documentation! — verl documentation](https://verl.readthedocs.io/en/latest/index.html).

3. **Install Required Packages**

   ```bash
   pip install -r requirement.txt
   ```

## Data Preparation

Prepare the necessary datasets before training with RPG. We use **filtered DAPO-Math-17k** dataset for fine-tuning and **AIME2024, AIME2025, AMC23** for evaluation. All the dataset should be pre-processed into .parquet form with the following pandas data-frame template and put in "data/" folder.

| Key         | data_source                                                  | prompt                                                       | ability | reward_model                                                 | extra_info                                                   |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Description | the source type of data, which decides the evaluation way for the rollout results. | Dict: {"content": The question embedded by the prompt template for the model, "role": "user"} | MATH    | Dict: {"ground_truth": string of the result, "style": "rule-lighteval/MATH_v2"} | Dict: {"index": the repeat index for acc@k, "raw_problem": the original question before embedding into the prompt template, "split": None} |

  - For the "content" in "prompt", we use the following prompt function:

    ```python
    def generate_prompt(question: str):
        pre_q = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        post_q = "\n\nRemember to put your answer on its own line after \"Answer:\"."
        return pre_q+question+post_q
    ```

  - For "data_source", we use

    - "math-dapo" for AIME2024 and DAPO-Math-17k
    - "aime2025" for AIME2025
    - "amc_23" for AMC-23

  - For "index" of "extra_info"

    - Accuracy@k is available by repeating the items in your data file for k times, and setting the "index" to 0~(k-1) for the k different copies of each item.

  - We provide the **AMC-23, AIME2024 and AIME2025** dataset in "data/" folder, and the data file for **filtered DAPO-Math-17k** dataset is available in https://huggingface.co/datasets/math-dataset/DAPO-17k-Eng/blob/main/dapo-math-17k-eng.parquet

## Repeat the Experiments

  1. Launch ray server.

     1. Using "ray start --head --dashboard-host=0.0.0.0" for dashboard and "ray start --address='YOUR_IP_ADDRESS:6379'" to launch ray server, where YOUR_IP_ADDRESS should be replaced by your local IP address

  2. Then run the scripts in "recipe/rpg/7b/" folder for reproducing experiments in the paper.

     - The scripts may be named as "run_A_7b_BCD.sh"

       - A is dapo/grpo/rfpp/rfppb for **baseline experiments**, where B is void string;
       - Or A is rpg, then
         - B is fkl/rkl/ufkl/urkl + 1e-4_ for **RPG experiments**
         - or B is fkl/rkl/ufkl/urkl + 1e-4_rf_clip_ppo_ + 0.2_0.28\_/0.1_0.1_ for **RPG-REINFORCE experiments** with clip parameters (0.2, 0.28) or (0.1, 0.1)
       - C is void string for AdamW optimizer or schedule_free for Schedule-free optimizer

     - For example, using

       ```
       bash recipe/rpg/7b/run_rpg_7b_fkl1e-4_rf_clip_ppo_0.1_0.1_7binstruct.sh
       ```

       for running RPG-REINFORCE-FKL experiments with AdamW optimizer and Qwen-2.5-7B-Instruct model with clip parameters (0.1, 0.1)

## Acknowledgements

  - [volcengine/verl: verl: Volcano Engine Reinforcement Learning for LLMs](https://github.com/volcengine/verl) for providing coding base
  - [Hugging Face](https://huggingface.co/) for providing the datasets:
    - [Mathematical Association of America (MAA)](https://artofproblemsolving.com/wiki/index.php/Mathematical_Association_of_America) for providing [AMC-23](https://artofproblemsolving.com/wiki/index.php/AMC_12_Problems_and_Solutions) and [AIME](https://artofproblemsolving.com/wiki/index.php/American_Invitational_Mathematics_Examination) datasets
    - [BytedTsinghua-SIA](https://air.tsinghua.edu.cn/en/About_Us/About_AIR.htm) for providing [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=complex-reasoning/RPG&type=Date)](https://star-history.com/#complex-reasoning/RPG&Date)

## Citation

If you use Regularized Policy Gradient (RPG) in your research or application, please consider citing it!

```bibtex
@article{zhang2025design,
    title={On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning},
    author={Zhang, Yifan and Liu, Yifeng and Yuan, Huizhuo and Yuan, Yang and Gu, Quanquan and Yao, Andrew C},
    journal={arXiv preprint arXiv:2505.17508},
    year={2025},
}
```
