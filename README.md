# Task-1# Step 1: Install necessary libraries
!pip install transformers datasets accelerate --quiet

# Step 2: Import required modules
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
from datasets import Dataset
from google.colab import files
import os

# Step 3: Load tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Fix for padding token
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Step 4: Upload your custom dataset (train.txt)
print("Please upload your train.txt file...")
uploaded = files.upload()

# Step 5: Read and prepare data
with open("train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

data = {"text": [line.strip() for line in lines if len(line.strip()) > 0]}
dataset = Dataset.from_dict(data)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 6: Set training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
)

# Step 7: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Step 8: Train the model
trainer.train()

# Step 9: Save model and tokenizer
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

# Step 10: Generate text
print("\nText Generation Output:")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Once upon a time"
output = generator(prompt, max_length=100, num_return_sequences=1)
print(output[0]["generated_text"])
