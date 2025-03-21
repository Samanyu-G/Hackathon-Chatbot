from typing import Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

def train_personality_model(personality_name, train_file):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=f"./{personality_name}_model",  # Save the model in a directory
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,  
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(f"./{personality_name}_model")
    tokenizer.save_pretrained(f"./{personality_name}_model")  # Save the tokenizer
#example_usage
train_personality_model("chef_antonio", "chef_data.csv")
train_personality_model("professor_amelia", "teacher_data.csv")
train_personality_model("bollywood_actor_raj", "bollywood actor_data.csv")
train_personality_model("banker_morgan", "banker_data.csv")