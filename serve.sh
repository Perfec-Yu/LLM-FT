TF_CPP_MIN_LOG_LEVEL=3 python -m EasyLM.models.llama.llama_serve \
    --mesh_dim='1,-1,1' \
    --load_llama_config='7b' \
    --load_checkpoint='params::gs://pengfei4/logs_vm2/5629a019fcca44da8f4189d8cac9b67e/streaming_params' \
    --tokenizer.vocab_file='gs://pengfei4/models/tokenizer.model' \
    --lm_server.chat_user_prefix=$'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n' \
    --lm_server.chat_lm_prefix=$'### Response:\n' \
    # --dtype=bf16