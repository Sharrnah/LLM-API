# Installation

Place compatible LLAMA2 AI model in gguf format into `cache/llama2/` directory.

for example one of here: https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF#provided-files

For summary support,
git clone https://huggingface.co/kabita-choudhary/finetuned-bart-for-conversation-summary into `cache/summary/` directory.

Build the Dockerimage using
```sh
docker build . -t llm-api-llama_cpp
```

Run the Dockerimage using (replace ##SOME_TOKEN## with your token you want to secure the API endpoints with)
```sh
docker run -p 8001:8000 --restart=always --runtime=nvidia --gpus=all --env AUTH_TOKEN=##SOME_TOKEN## --detach  llm-api-llama_cpp
```

Call the API endpoint like this:
```sh
curl --request POST --url 'http://127.0.0.1:8001/chat?text_prompt=t=Hello%20how%20are%20you?' --header 'X-Auth-Token: ##SOME_TOKEN##'
```

# Todos
- implement Vector Database for long time memory
