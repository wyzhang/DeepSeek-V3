import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Define model name
local_model_path = "/home/wyzhang/deepseek-moe-16b-base"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=True
)

# Load model (automatically places on GPU)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Ensure padding token exists
model.generation_config = GenerationConfig.from_pretrained(local_model_path)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

print("DeepSeek 16B Model Loaded Successfully!")

texts = [
    "Artificial intelligence is",
    "Explain what AI is"
    "What is deepseek"
]

for text in texts:
    input = tokenizer(text, return_tensors="pt").to("cuda")  # Move to GPU
    # Generate output
    output = model.generate(**input, max_new_tokens=100)
    # Decode and print result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'input text: ${text}')
    print(f'output text: ${result}')

