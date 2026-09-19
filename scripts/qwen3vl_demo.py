#export HF_ENDPOINT=https://hf-mirror.com

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from transformers import AutoProcessor, BitsAndBytesConfig
import json


try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    #quantization_config=quantization_config,
    device_map="auto",
)

model.eval()

question = "What is happening in the image?"

answer_a = "A person is holding candies."
answer_b = "A person is holding colorful balls."

judge_prompt = f"""
You are a multimodal judge.

Question:
{question}

Candidate A:
{answer_a}

Candidate B:
{answer_b}

Judge the two candidate answers using the image.
First output exactly A or B.
Then give a short reason.
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/mnt/dataset3/hongjun/ERMJ/candy.JPG",
            },
            {
                "type": "text",
                "text": judge_prompt,
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

print("Input shape:", inputs["input_ids"].shape)
print("Generated sequence shape:", outputs.sequences.shape)
print("Number of score steps:", len(outputs.scores))

generated_ids = [
    output_ids[input_ids.shape[-1]:]
    for input_ids, output_ids in zip(
        inputs["input_ids"],
        outputs.sequences,
    )
]

answers = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

answer = answers[0].strip()

print("Generated token ids:", generated_ids[0].tolist())
print("Model answer:")
print(repr(answer))

entropy_trajectory = []

for score in outputs.scores:
    #[batch_size, vocab_size]
    logits = score[0].float()

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    entropy = -(probs * log_probs).sum().item()
    entropy_trajectory.append(entropy)

result = {
    "prediction_text": answer,
    "entropy_trajectory": entropy_trajectory,
    "entropy_mean": (
        sum(entropy_trajectory) / len(entropy_trajectory)
        if entropy_trajectory
        else None
    ),
    "entropy_max": max(entropy_trajectory)
    if entropy_trajectory
    else None,
    "num_generated_tokens": len(entropy_trajectory),
}


print("Number of generated steps:", len(entropy_trajectory))
print("Entropy trajectory:", entropy_trajectory)

debug_dir = "/mnt/dataset3/hongjun/ERMJ/results/debug_qwen3vl.json"
with open(
    debug_dir,
    "w",
    encoding="utf-8",
) as f:
    json.dump(result, f, ensure_ascii=False, indent=2)