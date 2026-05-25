def init_agent(kwargs):
    if kwargs["method"] == "RuleNeSy":
        from .nesy_agent.rule_driven_rec import RuleDrivenAgent

        agent = RuleDrivenAgent(
            env=kwargs["env"],
            backbone_llm=kwargs["backbone_llm"],
            cache_dir=kwargs["cache_dir"],
            debug=kwargs["debug"],
        )
    elif kwargs["method"] == "LLMNeSy":
        from .nesy_agent.llm_driven_rec import LLMDrivenAgent

        agent = LLMDrivenAgent(
            **kwargs
        )
    elif kwargs["method"] == "Act":
        from .pure_neuro_agent.pure_neuro_agent import ActAgent
        from .pure_neuro_agent.prompts import ZEROSHOT_ACT_INSTRUCTION

        agent = ActAgent(
            env=kwargs["env"],
            backbone_llm=kwargs["backbone_llm"],
            prompt=ZEROSHOT_ACT_INSTRUCTION,
        )
    elif kwargs["method"] == "ReAct":
        from .pure_neuro_agent.pure_neuro_agent import ReActAgent
        from .pure_neuro_agent.prompts import (
            ONESHOT_REACT_INSTRUCTION,
            ONESHOT_REACT_INSTRUCTION_GLM4,
        )

        agent = ReActAgent(
            env=kwargs["env"],
            backbone_llm=kwargs["backbone_llm"],
            prompt=(
                ONESHOT_REACT_INSTRUCTION
                if "glm4" not in kwargs["backbone_llm"].name.lower()
                else ONESHOT_REACT_INSTRUCTION_GLM4
            ),
        )
    elif kwargs["method"] == "ReAct0":
        from .pure_neuro_agent.pure_neuro_agent import ReActAgent
        from .pure_neuro_agent.prompts import (
            ZEROSHOT_REACT_INSTRUCTION,
            ZEROSHOT_REACT_INSTRUCTION_GLM4,
        )

        agent = ReActAgent(
            env=kwargs["env"],
            backbone_llm=kwargs["backbone_llm"],
            prompt=(
                ZEROSHOT_REACT_INSTRUCTION
                if "glm4" not in kwargs["backbone_llm"].name.lower()
                else ZEROSHOT_REACT_INSTRUCTION_GLM4
            ),
        )
    elif kwargs["method"] == "LLM-modulo":
        from .nesy_verifier import LLMModuloAgent

        kwargs["model"] = kwargs["backbone_llm"]
        kwargs["max_steps"] = kwargs["refine_steps"]
        agent = LLMModuloAgent(
            **kwargs
        )
    elif kwargs["method"] == "TPCAgent":
        from .tpc_agent.tpc_agent import TPCAgent

        agent = TPCAgent(
            **kwargs
        )
    else:
        raise Exception("Not Implemented")
    return agent


def init_llm(llm_name, max_model_len=None):
    from .llms import Deepseek, GPT4o, GPT4oMini, GLM4Plus, Qwen, QwenAPI, Mistral, Llama, EmptyLLM

    from .tpc_agent.tpc_llm import TPCLLM

    if llm_name == "deepseek":
        llm = Deepseek()
    elif llm_name == "gpt-4o":
        llm = GPT4o()
    elif llm_name == "gpt-4o-mini":
        llm = GPT4oMini()
    elif llm_name == "glm4-plus":
        llm = GLM4Plus()
    elif llm_name == "qwen3-8b":
        # 远程 qwen3-8b，通过 OpenAI 兼容网关（默认 DMXAPI）调用
        llm = QwenAPI()
    elif "Qwen" in llm_name:
        # 本地 vLLM 推理（目录名中需包含 Qwen）
        llm = Qwen(llm_name, max_model_len=max_model_len)
    elif llm_name == "mistral":
        llm = Mistral(max_model_len=max_model_len)
    elif "Llama" in llm_name:
        llm = Llama(llm_name)
    elif llm_name == "rule":
        return EmptyLLM()
    elif llm_name == "TPCLLM":
        llm = TPCLLM()
    else:
        raise Exception("Not Implemented")

    return llm
