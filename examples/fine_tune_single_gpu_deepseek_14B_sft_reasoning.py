from llm_recovery.load_llm.load_llm import LoadLLM
from llm_recovery.fine_tuning.lora import LORA
import llm_recovery.constants.constants as constants
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
