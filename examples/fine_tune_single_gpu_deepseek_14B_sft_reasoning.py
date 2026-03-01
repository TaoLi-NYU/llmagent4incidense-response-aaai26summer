from llm_recovery.load_llm.load_llm import LoadLLM
"""

LoadLLM
来自 llm_recovery.load_llm.load_llm
👉 封装好的“加载基础大模型”的工具
内部帮你做好：
调用
“AutoModelForCausalLM.from_pretrained(...)”

“AutoTokenizer.from_pretrained(...)”

以及各种 device_map / 量化配置

“最常见的几个 Auto 系列”
🌱 纯 “底座” 型
这些只给你 encoder/decoder 本体，不带任务 head
    AutoModel
加载 基础 encoder/decoder, 比如 BERT 的 encoder、RoBERTa encoder
输出的是 hidden states, 用来自己接下游头
    AutoModelForCausalLM
你已经在用的这个：自回归语言模型，适合 chat / SFT / generation
GPT、LLaMA、Qwen、DeepSeek 这类都走它
    AutoModelForSeq2SeqLM
Encoder-decoder 结构，做 翻译 / summarization / 通用 seq2seq 生成，比如 T5、BART

1. AutoModelForCausalLM 是谁？
    AutoModelForCausalLM 是 transformers 里的一个 “自动选择模型类的工厂”。
你只提供一个名字 / 路径，比如 "gpt2", "meta-llama/Llama-2-7b-hf", "kimhammar/LLMIncidentResponse"
它会“自动判断”：
这个模型应该用 GPT2LMHeadModel、LlamaForCausalLM 还是别的什么具体类；
然后帮你构造 适合做因果语言建模(Causal LM) 的模型实例。

   所以与其自己写：
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
不如写：
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
未来换模型架构也“不用改类名，只改字符串”就行


2.  .from_pretrained(...) 具体做了几件事？
2.a     下载 / 读取配置(config)
去 HuggingFace Hub 上找 kimhammar/LLMIncidentResponse 这个 repo  (仓库repository, repo)
或者如果你传的是本地路径，比如 "./llm_ir_checkpoint"，就读本地；
载入 config.json, 里面写着:
模型类型（"model_type": "llama" / "qwen" 等）
hidden size、层数、attention heads 数量等等。
2.b     根据 config 选对模型类并创建空架构
AutoModelForCausalLM 看到 model_type = "llama"，就知道应该用 LlamaForCausalLM
于是先构建一个空的 Llama 模型架构(随机初始化)
2.c     加载预训练参数(权重)
读取模型权重文件(pytorch_model.bin / model.safetensors 等)
把里面的张量加载到刚才那个模型架构里；
最终返回一个已经带好权重的模型，可以直接用来推理或继续微调

3. AutoTokenizer 是谁？
AutoTokenizer 和 AutoModel... 很像，也是一个 “自动选择分词器类的工厂”
不同模型有不同的 tokenizer:
BPE、WordPiece、SentencePiece、tiktoken 风格……
你只需要写：
tokenizer = AutoTokenizer.from_pretrained("kimhammar/LLMIncidentResponse")
它会自动根据仓库里的配置文件(tokenizer_config.json, tokenizer.json 等)选择正确的 tokenizer 类

4. .from_pretrained(...) 做了哪些事？
4.a     找到并读取 tokenizer 配置
读取 tokenizer_config.json / special_tokens_map.json 等；
里面写了：
vocab 文件位置；
分词算法类型；
特殊 token(pad_token_id, bos_token_id, eos_token_id, unk_token_id……)
4.b     读取词表 / merge 规则 / SentencePiece 模型
对于 BPE: 会加载 vocab.json + merges.txt
对于 SentencePiece: 会加载 .model 文件；
最终构建一个能「字符串 ↔ token id」互相转换的对象

4.b Ⅰ 
BPE (Byte Pair Encoding)
核心思想（非常经典）：
从小粒度开始（字符/字节），统计最常一起出现的 pair, 不断合并, 学出一套子词。
简单版流程（假想英文）：
一开始把单词当成字符序列：
"hello" → h e l l o
统计所有文本里出现频率最高的相邻字符对，比如 ("l","l"), ("e","r") 之类
选一个频率最高的 pair 合并成新 token:
发现 ("l","l") 很频繁 → 新 token "ll"
"hello" → h e ll o
重复：再统计 pair, 合并…
发现 "he" 很多 → 合成 "he"
"hello" → he ll o
不断合并，直到达到预设词表大小(比如 32k)

缺点：
对空格、词边界的处理需要特判(比如 Ġword 这样的前缀 token)
主要针对欧语，对中文/混合文本不够优雅

4.b Ⅱ 
SentencePiece 是 Google 出的一个库/框架，支持多种分词算法，但最有名的是它的 Unigram model
对于 “Unigram”,大致流程是:

4.c     构建 tokenizer 实例
返回一个 tokenizer 对象，你可以：
tokenizer("hello") 得到 input_ids
tokenizer.decode(ids) 把 id 还原成字符串

"""


from llm_recovery.fine_tuning.lora import LORA
"""
llm_recovery: 这个项目的顶层包(你 pip install -e . 之后就有了)
llm_recovery.fine_tuning: 包下面的一个子包, 专门放“微调相关代码”
llm_recovery.fine_tuning.lora: 子包里的一个模块文件, 名字叫 lora.py。
LORA: lora.py 这个文件里定义的一个 类(Class)

导入之后，你在脚本里就可以直接写：
LORA.setup_llm_for_fine_tuning(...)
LORA.supervised_fine_tuning(...)
而不用写很长的 llm_recovery.fine_tuning.lora.LORA

from llm_recovery.fine_tuning.lora import LORA
⇒ 把 LoRA 微调工具类引进来。
LORA.setup_llm_for_fine_tuning(...)
⇒ 在基础大模型上插 LoRA adapter, 让它变成“可微调版本”。
LORA.supervised_fine_tuning(...)
⇒ 用你的 incident-response 数据集，在单 GPU 上跑 LoRA 微调
"""

import llm_recovery.constants.constants as constants
"""
等价于from llm_recovery.constants import constants
把这个模块在当前文件里简称为 constants, 然后用 constants.XXX 来访问

llm_recovery:这个项目的顶层包(你已经 pip install -e . 了)
llm_recovery.constants:包下面的一个子包,名字叫 constants
llm_recovery.constants.constants: 子包里的一个模块文件,通常就是
llm_recovery/constants/constants.py 这个文件
"""

from llm_recovery.fine_tuning.examples_dataset import ExamplesDataset
from transformers import set_seed
from datasets import load_dataset

if __name__ == '__main__':
    # 表示：只有当你 直接运行这个 py 文件 时，这块代码才会执行；如果是被别的文件 import，就不会跑这段（防止重复执行训练）
    seed = 99125
    set_seed(seed)
    """
    CPU随机性主要来自:
    Python 内置 random
    NumPy 的 numpy.random
    PyTorch CPU 端: torch.manual_seed 控制的 CPU RNG
    
    CPU上生成随机数, 比如:
    初始化一个 CPU tensor: torch.randn(3, 4) (模型在 CPU 时)
    用 NumPy 生成噪声: np.random.randn(...)

    GPU:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)(多GPU)

    transformers 的 set_seed(seed):
    HuggingFace 帮你封装好的“一键设随机种子”
    set_seed(seed) ≈ 帮你同时给 Python / NumPy / torch CPU+CUDA 都设 seed
    """
    device_map = {"": 0}
    
    """
    device_map
    每一份模型装在哪里
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "xxx/your-model",
        device_map={"": 0},  # "" 表示“整个模型”, 0 表示 GPU0
    )
    device_map = "auto"  
    # 这时候：
    如果你有多块 GPU, 它会按内存情况分层分散
    如果显存不够，它可能把一部分层放在 CPU / GPU 混合

    device_map = {
        "model.embed_tokens": 0,    #  embedding嵌入,用gpu0 
        "model.layers.0": 0,    # 放在gpu0
        "model.layers.1": 0,  
        "model.layers.2": 1,
        "model.layers.3": 1,
        "lm_head": 1,  #输出层,language modeling head 放在gpu1
    }
        embed_tokens.weight.shape == (10000, 768)
    可以理解成一个表：
    有  10000 行，每一行对应一个 token(比如 [PAD], I, am, hungry, ##ing, …)
    有 768 列，表示这个 token 在 768 维空间里的坐标

    """
    tokenizer, llm = LoadLLM.load_llm(llm_name=constants.LLM.DEEPSEEK_1_5B_QWEN, device_map=device_map)
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files="examples_16_june.json")
    instructions = dataset["train"]["instructions"][0]
    answers = dataset["train"]["answers"][0]
    """
    dataset:类型是 DatasetDict, 可以看成「好几个表的字典」, 键一般是 "train", "test", "validation"
    dataset["train"]：就是其中一张表 → 类型是 datasets.Dataset
    DatasetDict ≈ {"train": Dataset, "validation": Dataset, ...}
    Dataset ≈ 一张带多列的表(每列是一种字段)
    
    dataset["train"]["instructions"]   # 形如 [ ["instr0", "instr1", ...] ]
    dataset["train"]["instructions"][0]  # → ["instr0", "instr1", ...]
    
    answers = dataset["train"]["answers"][0] 同理
    
    这三行的结果就是拿到两条等长的 Python 列表:
    instructions = ["指令1", "指令2", "指令3", ...]
    answers      = ["答案1", "答案2", "答案3", ...]

    """
    lora_rank = 64
    lora_alpha = 128
    lora_dropout = 0.05
    llm = LORA.setup_llm_for_fine_tuning(llm=llm, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    dataset = ExamplesDataset(instructions=instructions, answers=answers, tokenizer=tokenizer)
    lr = 0.00095
    per_device_batch_size = 1  # 5
    num_train_epochs = 2
    prompt_logging_frequency = 50
    max_generation_tokens = 256 # 6000
    logging_steps = 1
    running_average_window = 100
    temperature = 0.6
    save_steps = 25
    save_limit = 3
    gradient_accumulation_steps = 1 # 16
    progress_save_frequency = 10
    LORA.supervised_fine_tuning(llm=llm, dataset=dataset, learning_rate=lr,
                                per_device_train_batch_size=per_device_batch_size,
                                num_train_epochs=num_train_epochs, logging_steps=logging_steps, prompts=instructions,
                                answers=answers,
                                prompt_logging=True,
                                running_average_window=running_average_window,
                                max_generation_tokens=max_generation_tokens,
                                prompt_logging_frequency=prompt_logging_frequency, temperature=temperature,
                                save_steps=save_steps, save_limit=save_limit,
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                progress_save_frequency=progress_save_frequency, seed=seed)
