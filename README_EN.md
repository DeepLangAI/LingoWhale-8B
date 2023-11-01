<p align="left">
    English</a>&nbsp ｜ &nbsp<a href="README.md">中文</a>
</p>
<br>

<div align="center">
<h1>
LingoWhale-8B
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/deeplang-ai/LingoWhale-8B" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/models/DeepLang/LingoWhale-8B" target="_blank">ModelScope</a> • ⛵ <a href="https://wisemodel.cn/models/%E5%8C%97%E4%BA%AC%E6%B7%B1%E8%A8%80%E7%A7%91%E6%8A%80%E6%9C%89%E9%99%90%E8%B4%A3%E4%BB%BB%E5%85%AC%E5%8F%B8/LingoWhale-8B/" target="_blank">Wisemodel</a>
</p>

<div align="center">
<strong>
LingoWhale-8B model open-sourced by DeepLangAI in collaboration with THUNLP Lab 🎉
</strong>
</div>

# Table of Contents

- [Introduction](#introduction)
- [Evaluation](#evaluation)
- [Generated Examples](#generated-examples)
- [Deployment and Inference](#deployment-and-inference)
- [Fine-tuning](#fine-tuning)
- [Open Source License](#open-source-license)

# Introduction

LingoWhale-8B is the first open-source model in the LingoWhale series introduced by DeepLangAI. It's a bilingual (Chinese-English) large language model.

LingoWhale-8B has been pre-trained on a large volume of high-quality bilingual data and exhibits powerful capabilities as a foundation model. It has achieved leading results on multiple public benchmarks. During its pre-training phase, the model was trained with a context window of 8K, allowing it to comprehend and generate longer sequences.

LingoWhale-8B is fully open for academic research. Users can apply for commercial use by email, and once granted official commercial permission, they can use it for free.

Along with open-sourcing the model weights, we also provide a Huggingface inference interface and parameter efficient fine-tuning examples like LoRA, making it easier for developers to use the LingoWhale-8B model.

Due to the scale of model parameters, intrinsic issues of large language models like hallucination and relatively weak mathematical computation capabilities persist in LingoWhale-8B. Please understand these issues and evaluate the possible risks before using the model. Future versions of the LingoWhale model will focus on optimizing these areas.

# Evaluation

We tested on the following public evaluation benchmarks:

- [C-Eval](https://arxiv.org/abs/2305.08322) is a Chinese foundation model evaluation benchmark consisting of 13,948 multiple-choice questions, covering 52 different subjects and four difficulty levels. It aims to assess the capability of Chinese language models. We used the dev set of this dataset as a few-shot source and conducted a 5-shot test on the test set.
  
- [MMLU](https://arxiv.org/abs/2009.03300) is an English foundation model evaluation benchmark that spans various domains like basic mathematics, American history, computer science, law, among others, with a total of 57 tasks. It evaluates language models' performance on different domain tasks. We performed a 5-shot test on this benchmark.
  
- [CMMLU](https://arxiv.org/abs/2306.09212) is a Chinese evaluation benchmark that encompasses 67 topics ranging from basic subjects to advanced professional levels. It evaluates Chinese language models' performance in knowledge and reasoning capabilities. We used the dev set of this dataset as a few-shot source and conducted a 5-shot test on the test set.
  
- [Gaokao](https://arxiv.org/abs/2305.12474) is an evaluation benchmark based on the dataset of Chinese college entrance examination questions. It aims to provide an assessment of Chinese language models in terms of language comprehension and logical reasoning capabilities. We retained only the four-option multiple-choice questions from it and conducted a 5-shot test after random partitioning.
  
- [HumanEval](https://arxiv.org/abs/2107.03374) is an English evaluation benchmark containing over one hundred coding problems. It assesses language models' abilities in code comprehension and generation. We adopted a zero-shot setting and the Pass@1 metric for testing the model.
  
- [GSM8K](https://arxiv.org/abs/2110.14168) is a dataset composed of high-quality elementary school math application problems. It requires the models to select the most appropriate solution based on the provided scenario and evaluates the models' capabilities in mathematical application. We conducted an 8-shot test on this benchmark.
  
- [BBH](https://arxiv.org/abs/2210.09261) is an evaluation benchmark formed from a selection of challenging tasks out of 204 Big-Bench benchmark tasks. We performed a 3-shot test on this benchmark.
  
- [AGIEval](https://arxiv.org/abs/2304.06364) is a benchmark to examine foundation models' human-like capabilities, specifically assessing foundational models' abilities in human cognition and problem-solving tasks. We retained only the four-option multiple-choice questions from it and conducted a 5-shot test after random partitioning.
  
These evaluation benchmarks provide standardized tests and metrics to assess language models' performance and capabilities across various tasks and domains. The evaluation results are shown in the table below:

| **Model**              |**C-Eval**| **MMLU** |**CMMLU** |**GAOKAO**|**HumanEval**|**GSM8K** | **BBH**  |**AGIEval**|
|:-----------------------|:--------:|:--------:|:--------:|:--------:|:-----------:|:--------:|:--------:|:---------:|
|                        |  5-shot  |  5-shot  |  5-shot  |  5-shot  |   0-shot    |  8-shot  |  3-shot  |  5-shot   |
| **GPT-4**              |   68.4   |   83.9   |   70.3   |   66.2   |    69.5     |   90.0   |   75.1   |   63.3    |
| **GPT-3.5 Turbo**      |   51.1   |   68.5   |   54.1   |   47.1   |    52.4     |   57.8   |   61.6   |   46.1    |
| **LLaMA2-7B**          |   28.9   |   45.7   |   31.4   |   26.0   |    12.8     |   16.2   |   39.2   |   26.5    |
| **ChatGLM2-6B**        |   50.2   |   45.9   |   49.0   |   49.4   |    9.2      |   28.9   |   31.7   |   45.3    |
| **Baichuan2-7B-Base**  |   54.0   |   54.2   |   57.1   |   47.5   |    18.3     |   24.5   |   41.6   |   42.7    |
| **Qwen-7B v1.1**       |   63.5   |   58.2   |   62.2   |     -    |    29.9     |   51.7   |   45.0   |     -     |
| **LingoWhale-8B-base** |   63.6   |   60.2   |   62.8   |   50.3   |    32.9     |   55.0   |   47.5   |   43.8    |

<span style="color:gray">For all of the above comparison models, we list their official reports and round them to one decimal place.</span>

# Generated Examples

The following examples are generated by a Chat model fine-tuned based on the LingoWhale-8B model using open source SFT data.

<details><summary><b>Mathematical calculations</b></summary><p>

**Query**：f(x)为偶函数， f(1) = 3, f(2) = 4, 那么f(1) + f(-2) ^2 等于多少？ 请一步一步说明

**Answer**：首先，根据题目中给出的条件，我们可以知道：

- f(x) 是一个偶函数，即 f(-x) = f(x)。
- f(1) = 3。
- f(2) = 4。

接下来，我们需要求出 f(1) + f(-2)^2 的值。
由于 f(x) 是偶函数，所以 f(-2) = f(2)，因此 f(-2) = 4。
将 f(1) 和 f(-2) 的值代入 f(1) + f(-2)^2 中，得到：
f(1) + f(-2)^2 = 3 + (4)^2 = 3 + 16 = 19。
因此，f(1) + f(-2)^2 的值为 19。

</details>

<details><summary><b>Role playing</b></summary><p>

**Query**：作为有丰富营销经验的品牌经理，请为我提供一条关于以下情境的品牌营销策略：
情境：618期间，提高品牌销售额
品牌定位：国产精品眉笔
建议要求：提供具有创新性和实效性的品牌营销策略，帮助提升品牌的销售额， 并加入直播带货的方案。

**Answer**：在618期间，为了提高品牌销售额，我们可以采取以下品牌营销策略：

1. 推出限时优惠活动：在618活动期间，我们可以在品牌官方网站上推出限时优惠活动，例如购买眉笔送赠品、打折促销等，以吸引消费者的注意力并促进销售。

2. 利用社交媒体宣传：我们可以通过社交媒体平台（如微博、微信、抖音等）发布品牌宣传内容，包括产品介绍、使用技巧、用户评价等，以增加品牌曝光率和用户互动性。

3. 与知名博主合作推广：我们可以邀请一些知名的美妆博主或网红来试用我们的产品，并在其社交媒体上发布推荐视频或图文，以吸引更多的目标受众。

4. 直播带货：我们可以邀请一些知名的带货主播或明星来为我们的产品进行直播带货，通过他们的影响力来吸引更多的消费者关注和购买。

5. 参与电商平台活动：我们可以积极参与各大电商平台的促销活动，例如满减、满赠等，以吸引更多的消费者购买我们的产品。

综上所述，以上品牌营销策略可以帮助我们在618期间提高品牌销售额，同时加入直播带货的方案可以更好地吸引目标受众，提高销售效果。

</details>

<details><summary><b>Poetry Writing</b></summary><p>

**Query**：写一个七言诗，描述岳飞的诗。诗词希望是宋代诗的风格。

**Answer**：《岳飞》

铁马金戈壮士魂，

精忠报国志凌云。

千古英雄谁可及，

岳家军威震乾坤。

</details>

<details><summary><b>Cultural knowledge</b></summary><p>

**Query**：我要去成都武侯祠参观，能帮我详细介绍下武侯的生平事迹吗？

**Answer**：诸葛亮（181年-234年），字孔明，号卧龙，徐州琅琊阳都（今山东临沂市沂南县）人，三国时期蜀汉丞相，杰出的政治家、军事家、文学家、书法家、发明家。早年随叔父诸葛玄到荆州，诸葛玄死后，诸葛亮就在隆中隐居。刘备三顾茅庐请出诸葛亮，联合东吴孙权于赤壁之战大败曹军，形成三国鼎足之势，又夺占荆益。刘备入蜀后，诸葛亮镇守荆州。刘备称帝后，诸葛亮被任命为丞相，主持朝政。后主刘禅继位，诸葛亮被封为武乡侯，领益州牧。勤勉谨慎，大小政事必亲自处理，赏罚严明；与东吴联盟，改善和西南各族的关系；实行屯田政策，加强战备。前后五次北伐中原，多以粮尽无功。终因积劳成疾，于234年病逝于五丈原（今陕西岐山县南），享年54岁。刘禅追封其为忠武侯，后世常以武侯尊称诸葛亮。东晋政权因其军事才能特追封他为武兴王。诸葛亮散文代表作有《出师表》、《诫子书》等。曾发明木牛流马、孔明灯等，并改造连弩，叫做诸葛连弩，可一弩十矢俱发。诸葛亮一生“鞠躬尽瘁、死而后已”，是中国传统文化中忠臣与智者的代表人物。

</details>

<details><summary><b>Code generation</b></summary><p>

**Query**：你是一名编程开发者，能写一个正则帮忙判断一个字符串是否是合理的手机号码， 用Python实现。

**Answer**：
```python
import re

def is_valid_phone(phone):
    pattern = r'^1[3-9]\d{9}$'
    return bool(re.match(pattern, phone))

print(is_valid_phone('13800138000'))  # True
print(is_valid_phone('12345678901'))  # False
```

</details>

# Deployment and Inference

## Install Dependencies
```shell
pip install -r requirements.txt
```

+ To use flash attention, you need to install [flash-attn](https://github.com/Dao-AILab/flash-attention)

+ For using LoRA or other parameter efficient fine-tuning methods, please install [peft](https://github.com/huggingface/peft)

## Python Example

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("deeplang-ai/LingoWhale-8B", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("deeplang-ai/LingoWhale-8B", device_map="auto", trust_remote_code=True)
>>> inputs = tokenizer("陋室铭\n唐 刘禹锡\n", return_tensors="pt")
>>> inputs = inputs.to("cuda:0")
>>> pred = model.generate(**inputs, max_new_tokens=100, repetition_penalty=1.1)
>>> print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

# Fine-tuning
The fine-tuning example is based on `transformers.Trainer`. For a more deteiled guide on the arguments usage, please refer to Huggingface [`Trainer`](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#trainer) tutorial.

The aim of this section is to showcase the fine-tuning process. No guarantees are made about the model performance under this fine-tuning configuration.

## Single-Machine Training
Below is an example of fine-tuning on a single machine. The data used is 10,000 randomly selected instruction fine-tuning data from the COIG dataset. You can replace it with your own data.

```shell
hostfile=""
deepspeed --hostfile=$hostfile finetune/finetune.py  \
    --report_to "none" \
    --data_path "finetune/data/coig_10k.json" \
    --model_name_or_path deeplang-ai/LingoWhale-8B \
    --output_dir "output" \
    --model_max_length 2048 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed finetune/ds_config.json \
    --bf16 True \
    --tf32 True
```

To use you own data, please convert it to the json format below
```json
[
    {
        "id": 0,
        "conversations": [
            {
                "from": "human",
                "value": "请问什么是“模式年龄”？"
            },
            {
                "from": "model",
                "value": "模式年龄是指利用放射性衰变规律假定地质样品形成时的初始同位素组成计算得到的年龄。"
            },
            ...
        ]
    },
    ...
]
```

## Multi-Machine Training

For multi-machine training, you need to create a `hostfile` in the following format. Each line represents a machine. `ip_address-X` refers to the IP address of each machine, and the `slots` content indicates the number of available GPUs on the machine. The content format is as follows:

```
ip_address-1 slots=8
ip_address-2 slots=8
ip_address-3 slots=8
ip_address-4 slots=8
...
```

Next, specify the `hostfile` argument using the path of the `hostfile`, and run the following command to start multi-machine training.

```shell
hostfile="/path/to/hostfile"
deepspeed --hostfile=$hostfile finetune/finetune.py  \
    --report_to "none" \
    --data_path "finetune/data/coig_10k.json" \
    --model_name_or_path deeplang-ai/LingoWhale-8B \
    --output_dir "output" \
    --model_max_length 2048 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed finetune/ds_config.json \
    --bf16 True \
    --tf32 True
```

## Parameter-Efficient Fine-Tuning
By using [peft](https://github.com/huggingface/peft), you can easily apply parameter-efficient fine-tuning methods like LoRA, Prefix-Tuning, etc. The training method for LoRA is currently integrated into the code, which can be activated by adding `--use_lora True`.

LoRA checkpoints can be loaded using the following code:
```python
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("output", trust_remote_code=True)
```

# Open Source License
The community use of the LingoWhale-8B model must adhere to the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) and the [LingoWhale-8B Model License Agreement](MODEL_LICENSE.md). If you wish to use this source model or its derivatives for commercial purposes, please refer to [LingoWhale-8B Model License Agreement](MODEL_LICENSE.md).
