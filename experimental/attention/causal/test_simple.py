from nnsight import LanguageModel
import torch

print("Loading model...")
model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")

# Test basic generation
simple_prompt = "Which is bigger: 9.8 or 9.11? Answer:"
qa_prompt = "Q: Which is bigger: 9.8 or 9.11? A:"

print("\nTesting Simple format:")
with model.generate(max_new_tokens=10, do_sample=False) as generator:
    with generator.invoke(simple_prompt):
        output_ids = model.generator.output.save()
output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Input: {simple_prompt}")
print(f"Output: {output}")

print("\nTesting Q&A format:")
with model.generate(max_new_tokens=10, do_sample=False) as generator:
    with generator.invoke(qa_prompt):
        output_ids = model.generator.output.save()
output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Input: {qa_prompt}")
print(f"Output: {output}")

print("\nTest complete!")