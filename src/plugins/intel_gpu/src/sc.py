import os

from transformers import AutoTokenizer, TextStreamer
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
import time
from openvino import Core

model_name = "microsoft/phi-2"

sample = """from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\""""




# Load tokenizer to be used with the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the sample
inputs = tokenizer([sample], return_tensors='pt')




core = Core()
print(core.get_versions('GPU'))

def run_stateful():
    print("------------stateful")
    save_name = model_name.split("/")[-1] + "_openvino"

    precision = "f32"
    quantization_config = OVWeightQuantizationConfig(
        bits=4,
        sym=False,
        group_size=128,
        ratio=0.8,
    )
    device = "gpu"

    # Load kwargs
    load_kwargs = {
        "device": device,
        "ov_config": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": precision,
            "CACHE_DIR": "",
        },
        "compile": False,
        "quantization_config": quantization_config
    }

    # Check whether the model was already exported
    saved = os.path.exists(save_name)

    model = OVModelForCausalLM.from_pretrained(
        model_name if not saved else save_name,
        export=not saved,
        **load_kwargs,
    )

    # Load tokenizer to be used with the model

    # Save the exported model locally
    if not saved:
        model.save_pretrained(save_name)

    # TODO Optional: export to huggingface/hub

    model_size = os.stat(os.path.join(save_name, "openvino_model.bin")).st_size / 1024 ** 3
    print(f'Model size in FP32: ~5.4GB, current model size in 4bit: {model_size:.2f}GB')


    model.compile()

    # Tokenize the sample
    inputs = tokenizer([sample], return_tensors='pt')

    # Call generate on the inputs
    start_regular = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        # streamer=TextStreamer(tokenizer=tokenizer, skip_special_tokens=True),
        pad_token_id=tokenizer.eos_token_id,
    )
    end_regular = time.time()

    return (start_regular, end_regular)

def run_pld():
    print("------------pld")
    # Save the model in a different directory to set it apart from the stateful model
    save_name = model_name.split("/")[-1] + "_openvino_stateless"

    precision = "f32"
    quantization_config = OVWeightQuantizationConfig(
        bits=4,
        sym=False,
        group_size=128,
        ratio=0.8,
    )
    device = "gpu"

    # Load kwargs
    load_kwargs = {
        "device": device,
        "ov_config": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": precision,
            "CACHE_DIR": "",
        },
        "compile": False,
        "quantization_config": quantization_config
    }

    # Check whether the model was already exported
    saved = os.path.exists(save_name)

    # We can use the same loading attributes, the only differece is the stateful attribute
    stateless_model = OVModelForCausalLM.from_pretrained(
        model_name if not saved else save_name,
        export=not saved,
        stateful=False,
        **load_kwargs,
    )

    # Save the exported model locally
    if not saved:
        stateless_model.save_pretrained(save_name)
        tokenizer.save_pretrained(save_name)

    stateless_model.compile()

    start_pld = time.time()
    out = stateless_model.generate(
        **inputs,
        max_new_tokens=128,
        # streamer=TextStreamer(tokenizer=tokenizer, skip_special_tokens=True),
        pad_token_id=tokenizer.eos_token_id,
        prompt_lookup_num_tokens=3,
    )
    end_pld = time.time()
    return (start_pld, end_pld)


def run_speculative():
    print("------------speculeative")
    model_name = "microsoft/phi-2"
    save_name = model_name.split("/")[-1] + "_openvino_stateless"
    precision = "f32"
    quantization_config = OVWeightQuantizationConfig(
        bits=4,
        sym=False,
        group_size=128,
        ratio=0.8,
    )
    device = "gpu"

    # Check whether the model was already exported
    saved = os.path.exists(save_name)

    device = "gpu"
    # Load kwargs
    load_kwargs = {
        "device": device,
        "ov_config": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": precision,
            "CACHE_DIR": "",
        },
        "compile": False,
        "quantization_config": quantization_config
    }

    # We can use the same loading attributes, the only differece is the stateful attribute
    stateless_model = OVModelForCausalLM.from_pretrained(
        model_name if not saved else save_name,
        export=not saved,
        stateful=False,
        **load_kwargs,
    )

    # Save the exported model locally
    if not saved:
        stateless_model.save_pretrained(save_name)

    print(stateless_model.ov_config)
    stateless_model.compile()


    model_name = "Salesforce/codegen-350M-multi"
    save_name = model_name.split("/")[-1] + "_openvino_stateless"
    precision = "f32"
    quantization_config = OVWeightQuantizationConfig(
        bits=4,
        sym=False,
        group_size=128,
        ratio=0.8,
    )
    device = "cpu"


    # Load kwargs
    load_kwargs = {
        "device": device,
        "ov_config": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": precision,
            "CACHE_DIR": "",
        },
        "compile": False,
        "quantization_config": quantization_config
    }


    asst_model = OVModelForCausalLM.from_pretrained(
        model_name if not saved else save_name,
        export=not saved,
        stateful=False,
        **load_kwargs,
    )

    # Save the exported model locally
    if not saved:
        asst_model.save_pretrained(save_name)

    print(asst_model.ov_config)
    asst_model.compile()


    asst_model.generation_config.num_assistant_tokens = 3
    asst_model.generation_config.num_assistant_tokens_schedule = 'const'

    start_speculative = time.time()
    out = stateless_model.generate(
        **inputs,
        max_new_tokens=128,
        # streamer=TextStreamer(tokenizer=tokenizer, skip_special_tokens=True),
        pad_token_id=tokenizer.eos_token_id,
        assistant_model=asst_model,
    )
    end_speculative = time.time()
    return (start_speculative, end_speculative, out)

start_regular = end_regular = start_pld = end_pld = 0

# start_regular, end_regular = run_stateful()
# start_pld, end_pld = run_pld()
start_speculative, end_speculative, out = run_speculative()

print("Regular %s ms ---" % ((end_regular - start_regular) * 1000))
print("PLD %s ms ---" % ((end_pld - start_pld) * 1000))
print("Speculative %s ms ---" % ((end_speculative - start_speculative) * 1000))
