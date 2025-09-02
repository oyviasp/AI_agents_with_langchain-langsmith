#https://www.youtube.com/watch?v=UtSSMs6ObqY
#start ollama app or "ollama serve" in terminal to get localhost port
#open terminal, ollama run "model name"
#/bye - out from chat
#ollama list - show all models



from langchain_ollama import ChatOllama
local_llm = "qwen2.5:7b"
model = ChatOllama(model=local_llm, temperature=0)

messages = [  
("system", "You are a helpful translator. Translate the user sentence to French."),  
("human", "I love programming."),  
]  

result = model.invoke(messages)
print(result.content)




# import ollama
# client = ollama.Client()
# model2 = "qwen2.5:7b"
# prompt = "Translate the following English text to French: 'I love programming.'"
# result = client.generate(model=model2, prompt=prompt)
# print(result["response"])