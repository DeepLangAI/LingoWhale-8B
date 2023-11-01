<p align="left">
    <a href="README_EN.md">English</a>&nbsp ｜ &nbsp中文
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
深言科技联合清华大学NLP实验室开源语鲸-8B模型 🎉
</strong>
</div>

# 目录

- [目录](#目录)
- [模型介绍](#模型介绍)
- [测评结果](#测评结果)
- [生成样例](#生成样例)
- [部署和推理](#部署和推理)
- [微调方法](#微调方法)
- [开源协议](#开源协议)

# 模型介绍

LingoWhale-8B是由深言科技推出的语鲸系列大模型中首个开源的中英双语大语言模型。

LingoWhale-8B模型在数万亿token的高质量中英数据上进行预训练，具有强大的基础能力，在多个公开评测基准上均达到领先效果。在预训练阶段，模型使用8K的上下文长度进行训练，能够完成更长上下文的理解和生成任务。

LingoWhale-8B模型对学术研究完全开放，使用方通过邮件申请并获得官方商用许可后，即可免费商用。

在开源模型权重的同时，我们也提供了符合用户习惯的Huggingface推理接口以及LoRA等参数高效微调示例，便于开发者快速使用LingoWhale-8B模型。

受模型参数量影响，大模型固有的幻觉问题、数学计算能力相对较弱等问题在LingoWhale-8B模型中仍然存在。请大家在使用前了解这些问题，评估可能存在的风险。后续版本的LingoWhale模型将会针对此类问题进行重点优化。


# 测评结果

我们在以下公开评测基准上进行了测试：

- [C-Eval](https://arxiv.org/abs/2305.08322)是一个中文基础模型评估基准，包含了13948个多项选择题，涵盖了52个不同的学科和四个难度级别。它旨在评估中文语言模型的能力。我们使用该数据集的dev集作为few-shot的来源，在test集上进行了5-shot测试。
- [MMLU](https://arxiv.org/abs/2009.03300)是一个英文基础模型评估基准，涵盖了基本数学、美国历史、计算机科学、法律等多个领域，共包含57个任务。它用于评估语言模型在不同领域任务上的表现。我们对模型进行了5-shot测试。
- [CMMLU](https://arxiv.org/abs/2306.09212)是一个中文评估基准，涵盖了从基础学科到高级专业水平的67个主题。它用于评估中文语言模型在知识和推理能力方面的表现。我们使用该数据集的dev集作为few-shot的来源，在test集上进行了5-shot测试。
- [Gaokao](https://arxiv.org/abs/2305.12474)是一个以中国高考题目为数据集的评估基准。它旨在提供测评中文语言模型在语言理解能力和逻辑推理能力方面的能力。我们只保留了其中的四选一的选择题，随机划分后对模型进行了5-shot测试。
- [HumanEval](https://arxiv.org/abs/2107.03374)是一个包含上百个编程问题的英文评估基准。它用于评估语言模型在程序理解与生成能力方面的表现。我们采用了zero-shot计算Pass@1的方法对模型进行了测试。
- [GSM8K](https://arxiv.org/abs/2110.14168)是一个由高质量、语言多样化的小学数学应用题组成的数据集。它要求根据给定的场景选择最合理的方案，用于评估语言模型在数学应用方面的能力。我们对模型进行了8-shot测试。
- [BBH](https://arxiv.org/abs/2210.09261)是一个从204项Big-Bench评测基准任务中选择出的表现较差的任务单独形成的评估基准。它用于评估大型语言模型在具有挑战性的任务上的表现。我们对模型进行了3-shot测试。
- [AGIEval](https://arxiv.org/abs/2304.06364)是一项考察基础模型类人能力的基准测试，专门用于评估基础模型在人类认知和问题解决相关任务中的能力。我们只保留了其中的四选一的选择题，随机划分后对模型进行了5-shot测试。


这些评估基准提供了标准化的测试和度量，用于评估语言模型在不同任务和领域上的性能和能力。评测方法和测评结果如下表所示：

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

<span style="color:gray">对于以上所有对比模型，我们列出了其官方汇报结果并四舍五入保留一位小数。</span>

# 生成样例

> 生成以下样例的模型是使用开源SFT数据基于LingoWhale-8B模型进行微调后的对话模型

<details><summary><b>数学计算</b></summary><p>

**问题**：f(x)为偶函数， f(1) = 3, f(2) = 4, 那么f(1) + f(-2) ^2 等于多少？ 请一步一步说明

**解答**：首先，根据题目中给出的条件，我们可以知道：

- f(x) 是一个偶函数，即 f(-x) = f(x)。
- f(1) = 3。
- f(2) = 4。

接下来，我们需要求出 f(1) + f(-2)^2 的值。
由于 f(x) 是偶函数，所以 f(-2) = f(2)，因此 f(-2) = 4。
将 f(1) 和 f(-2) 的值代入 f(1) + f(-2)^2 中，得到：
f(1) + f(-2)^2 = 3 + (4)^2 = 3 + 16 = 19。
因此，f(1) + f(-2)^2 的值为 19。

</details>

<details><summary><b>角色扮演</b></summary><p>

**问题**：作为有丰富营销经验的品牌经理，请为我提供一条关于以下情境的品牌营销策略：
情境：618期间，提高品牌销售额
品牌定位：国产精品眉笔
建议要求：提供具有创新性和实效性的品牌营销策略，帮助提升品牌的销售额， 并加入直播带货的方案。

**解答**：在618期间，为了提高品牌销售额，我们可以采取以下品牌营销策略：

1. 推出限时优惠活动：在618活动期间，我们可以在品牌官方网站上推出限时优惠活动，例如购买眉笔送赠品、打折促销等，以吸引消费者的注意力并促进销售。

2. 利用社交媒体宣传：我们可以通过社交媒体平台（如微博、微信、抖音等）发布品牌宣传内容，包括产品介绍、使用技巧、用户评价等，以增加品牌曝光率和用户互动性。

3. 与知名博主合作推广：我们可以邀请一些知名的美妆博主或网红来试用我们的产品，并在其社交媒体上发布推荐视频或图文，以吸引更多的目标受众。

4. 直播带货：我们可以邀请一些知名的带货主播或明星来为我们的产品进行直播带货，通过他们的影响力来吸引更多的消费者关注和购买。

5. 参与电商平台活动：我们可以积极参与各大电商平台的促销活动，例如满减、满赠等，以吸引更多的消费者购买我们的产品。

综上所述，以上品牌营销策略可以帮助我们在618期间提高品牌销售额，同时加入直播带货的方案可以更好地吸引目标受众，提高销售效果。

</details>

<details><summary><b>诗歌创作</b></summary><p>

**问题**：写一个七言诗，描述岳飞的诗。诗词希望是宋代诗的风格。

**解答**：《岳飞》

铁马金戈壮士魂，

精忠报国志凌云。

千古英雄谁可及，

岳家军威震乾坤。

</details>

<details><summary><b>文化知识</b></summary><p>

**问题**：我要去成都武侯祠参观，能帮我详细介绍下武侯的生平事迹吗？

**解答**：诸葛亮（181年-234年），字孔明，号卧龙，徐州琅琊阳都（今山东临沂市沂南县）人，三国时期蜀汉丞相，杰出的政治家、军事家、文学家、书法家、发明家。早年随叔父诸葛玄到荆州，诸葛玄死后，诸葛亮就在隆中隐居。刘备三顾茅庐请出诸葛亮，联合东吴孙权于赤壁之战大败曹军，形成三国鼎足之势，又夺占荆益。刘备入蜀后，诸葛亮镇守荆州。刘备称帝后，诸葛亮被任命为丞相，主持朝政。后主刘禅继位，诸葛亮被封为武乡侯，领益州牧。勤勉谨慎，大小政事必亲自处理，赏罚严明；与东吴联盟，改善和西南各族的关系；实行屯田政策，加强战备。前后五次北伐中原，多以粮尽无功。终因积劳成疾，于234年病逝于五丈原（今陕西岐山县南），享年54岁。刘禅追封其为忠武侯，后世常以武侯尊称诸葛亮。东晋政权因其军事才能特追封他为武兴王。诸葛亮散文代表作有《出师表》、《诫子书》等。曾发明木牛流马、孔明灯等，并改造连弩，叫做诸葛连弩，可一弩十矢俱发。诸葛亮一生“鞠躬尽瘁、死而后已”，是中国传统文化中忠臣与智者的代表人物。

</details>

<details><summary><b>代码生成</b></summary><p>

**问题**：你是一名编程开发者，能写一个正则帮忙判断一个字符串是否是合理的手机号码， 用Python实现。

**解答**：
```python
import re

def is_valid_phone(phone):
    pattern = r'^1[3-9]\d{9}$'
    return bool(re.match(pattern, phone))

print(is_valid_phone('13800138000'))  # True
print(is_valid_phone('12345678901'))  # False
```

</details>

# 部署和推理

## 安装依赖
```shell
pip install -r requirements.txt
```

+ 如需使用flash attention需要安装[flash-attn](https://github.com/Dao-AILab/flash-attention)
+ 如需使用LoRA等少参数微调方法需要额外安装[peft](https://github.com/huggingface/peft)

## Python 样例

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("deeplang-ai/LingoWhale-8B", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("deeplang-ai/LingoWhale-8B", device_map="auto", trust_remote_code=True)
>>> inputs = tokenizer("陋室铭\n唐 刘禹锡\n", return_tensors="pt")
>>> inputs = inputs.to("cuda:0")
>>> pred = model.generate(**inputs, max_new_tokens=100, repetition_penalty=1.1)
>>> print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

# 微调方法
模型微调样例代码以`transformers.Trainer`为基础，其中大部分参数和使用方法都可以参考Huggingface中[`Trainer`](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#trainer) 的教程和介绍。

> 本章节旨在展示微调过程，并不对该微调配置下进行微调后的模型效果进行保证。

## 单机训练
下面是一个单机进行微调的例子，使用的数据为从[COIG](https://huggingface.co/datasets/BAAI/COIG)数据集中随机选取的10000条指令微调数据，可以使用自己的数据进行替换。

```shell
hostfile=""
deepspeed --hostfile=$hostfile finetune/finetune.py  \
    --report_to "none" \
    --data_path "finetune/data/coig_10k.json" \
    --model_name_or_path deeplang-ai/LingoWhale-8B \
    --output_dir "output" \
    --model_max_length 2048 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
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

若要替换为自己的数据，可以使用如下格式的json文件。
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

## 多机训练

多机器训练需要编辑如下格式的`hostfile`文件。其中，每一行表示一个机器，`ip_address-X`为各个机器对应的ip地址，`slots`内容表示机器可用GPU数量。内容格式如下：

```
ip_address-1 slots=8
ip_address-2 slots=8
ip_address-3 slots=8
ip_address-4 slots=8
...
```

同时指定hostfile参数为`hostfile`文件路径，然后运行如下命令即可启动多机训练。

```shell
hostfile="/path/to/hostfile"
deepspeed --hostfile=$hostfile finetune/finetune.py  \
    --report_to "none" \
    --data_path "finetune/data/coig_10k.json" \
    --model_name_or_path deeplang-ai/LingoWhale-8B \
    --output_dir "output" \
    --model_max_length 2048 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
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
## 少参数微调
通过使用[peft](https://github.com/huggingface/peft)，可以轻松调LoRA, Prefix-Tuning等少参数微调的方法。目前在代码中集合了LoRA的训练方法，可以通过加入`--use_lora True`启动。

使用LoRA训练的checkpoint可以通过下面的代码读取和调用：
```python
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("output", trust_remote_code=True)
```

# 开源协议
社区使用LingoWhale-8B模型需要遵循[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)和[《LingoWhale-8B模型许可协议》](MODEL_LICENSE.md)。若您期望将此源模型或其衍生品用作商业用途，请参考[《LingoWhale-8B模型许可协议》](MODEL_LICENSE.md)。