from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "tiiuae/falcon-180B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = input("Enter you prompt : ")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    max_new_tokens=50,
)
output = output[0].to(device)
print(tokenizer.decode(output))
