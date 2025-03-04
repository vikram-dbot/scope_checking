from openai import OpenAI
import base64
openai = OpenAI(
    api_key="tigB029ct3yTUkqtvI1gNsdXtEJ8VqeR",
    base_url="https://api.deepinfra.com/v1/openai"
)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
image = "D:/Office/dataset_new/401-800/496.jpg"
image = encode_image(image)
chat_completion = openai.chat.completions.create(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct",
    messages = [{"role": "system", "content": "Analyze the image which is in base64 format to find the stage of construction it is in."},
        {"role": "user", "content": image}]
)
print(chat_completion.choices[0].message.content)
print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)