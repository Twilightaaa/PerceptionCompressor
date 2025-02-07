compression_rate=4
k_1=0.4
tau_o=0.2
k_2=0.1
coarse_control_coefficient=1
segment_size=200
save_folder_path=test
cnt_tokenizer_path=/path-to-dir/Meta-Llama-3-8B-Instruct
llama_path=/path-to-dir/llama2-7b-hf
bert_path=/path-to-dir/sentencebert
query_dict_path=/path-to-dir/Guiding_questions_for_NQ.json
qa_data_path=/path-to-dir/lost-in-the-middle/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl

CUDA_VISIBLE_DEVICES=0 python PerceptionCompressor.py \
    --compression_rate $compression_rate \
    --k_1 $k_1 \
    --tau_o $tau_o \
    --k_2 $k_2 \
    --coarse_control_coefficient $coarse_control_coefficient \
    --segment_size $segment_size \
    --save_folder_path $save_folder_path \
    --cnt_tokenizer_path $cnt_tokenizer_path \
    --llama_path $llama_path \
    --bert_path $bert_path \
    --query_dict_path $query_dict_path \
    --qa_data_path $qa_data_path