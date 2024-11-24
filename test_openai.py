import openai

openai.api_key = "sk-3gMvo3Rv6kwLD7LzXIPg T3BlbkFJ54R7PligQxcnhEAmSt67"

try:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Hello, OpenAI!",
        max_tokens=5
    )

    print(response.choices[0].text.strip())
except Exception as e:
    print(f"Error: {e}")
