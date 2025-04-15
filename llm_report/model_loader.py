from transformers import AutoTokenizer, AutoModelForCausalLM

# Cache is reused from ~/.cache/huggingface/
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
