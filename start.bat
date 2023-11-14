docker run -p 8001:8000 --restart=always --gpus=all --env AUTH_TOKEN=test -v $PWD/chat_histories:/app/chat_histories --detach llm-api-llama_cpp
