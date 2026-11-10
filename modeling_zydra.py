from transformers import LlamaConfig, LlamaForCausalLM


class ZydraConfig(LlamaConfig):
    model_type = "zydra"


class ZydraForCausalLM(LlamaForCausalLM):
    config_class = ZydraConfig


try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register(ZydraConfig.model_type, ZydraConfig)
    AutoModelForCausalLM.register(ZydraConfig, ZydraForCausalLM)
except ValueError:
    pass