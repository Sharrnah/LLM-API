docker run -p 8001:8000 --restart=always --runtime=nvidia --gpus=all --env AUTH_TOKEN=test -v $(pwd)/chat_histories:/app/chat_histories --detach  llm-api-llama_cpp
