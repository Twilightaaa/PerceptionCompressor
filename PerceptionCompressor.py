import json
import numpy as np
from icecream import ic as print
from itertools import zip_longest
import seaborn as sns
import math
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from sentence_transformers import util
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import argparse
import tiktoken
import httpx
from huggingface_hub import login
import os
from llmlingua import PromptCompressor

import time

import statistics

import pynvml


from openai import OpenAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t = "Question: {question}\nAnswer:"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def load_jsonl(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_condition_ppl(
    text: str,
    question: str,
    condition_in_question: str = "none",
    granularity: str = "sentence",
    model=None,
    tokenizer=None
):
    if condition_in_question == "none":
        return get_ppl(
            text, 
            granularity=granularity,
            model=model,
            tokenizer=tokenizer
        )
    elif condition_in_question == "before":
        return get_ppl(
            text=question + text,
            granularity=granularity,
            condition_mode="after",
            condition_pos_id=get_token_length(question, tokenizer) - 1,
            model=model,
            tokenizer=tokenizer
        )
    elif condition_in_question == "after":
        return get_ppl(
            text=text + question,
            granularity=granularity,
            condition_mode="after",
            condition_pos_id=get_token_length(text, tokenizer) - 1,
            model=model,
            tokenizer=tokenizer
        )

def get_ppl(
    text: str = "",
    granularity: str = "sentence",
    input_ids=None,
    attention_mask=None,
    past_key_values=None,
    return_kv=False,
    end=None,
    condition_mode: str = "none",
    condition_pos_id: int = 0,
    model=None,
    tokenizer=None
):
    if input_ids is None:
        tokenized_text = tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(device)
        attention_mask = tokenized_text["attention_mask"].to(device)

    past_length = 0
    if end is None:
        end = input_ids.shape[1]

    with torch.no_grad():
        response = model(
            input_ids[:, past_length:end],
            # attention_mask=attention_mask[:, :end],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = response.past_key_values

    shift_logits = response.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., past_length + 1: end].contiguous()

    active_logits = shift_logits.view(-1, shift_logits.size(-1))
    active_labels = shift_labels.view(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(active_logits, active_labels)
    if condition_mode == "before":
        loss = loss[:condition_pos_id]
    elif condition_mode == "after":
        loss = loss[condition_pos_id:]
    res = loss.mean() if granularity == "sentence" else loss
    return (res, past_key_values) if return_kv else res

def get_estimate_threshold_base_distribution(
    ppl, ratio: float = 0, remove_tokens: int = 0, reverse: bool = True
):      
    if (ratio == 0) and (remove_tokens == 0) and (reverse == True):
        return max(ppl) + 1
    elif (ratio == 0) and (remove_tokens == 0) and (reverse == False):
        return min(ppl) - 1
    
    if ratio != 0:
        target_token = max(0, min(len(ppl) - 1, int(len(ppl) * ratio) - 1))
    else:
        target_token = len(ppl) - 1 - remove_tokens
    
    threshold = sorted(ppl, reverse=reverse)[target_token].detach().cpu().item()

    return threshold

def get_token_length(
    text: str,
    tokenizer=None,
    add_special_tokens: bool = True,
):
    return len(
        tokenizer(text, add_special_tokens=add_special_tokens).input_ids
    )

def get_dynamic_compress_ratio(n=20, dynamic_context_compression_ratio=0.25):
    dynamic_ratio = [
        i * (abs(dynamic_context_compression_ratio) / (n - 1)) if n > 1 else 0 for i in range(-(n - 1), n, 2)
        ][::-1]
    return dynamic_ratio

def get_deltas(n, a, b):
    deltas = [b * (1 - 1.0 * i / ((1 - a) * n)) for i in range(1, n + 1, 1)]
    return deltas

def split_segments_input_ids(
    segments, 
    tokenizer, 
    args
):
    context = "\n".join(segments)
    raw_input_ids = tokenizer(context, return_tensors="pt").input_ids[0][1:]

    if len(raw_input_ids) % args.segment_size == 0:
        num_segments = len(raw_input_ids) // args.segment_size
    else:
        num_segments = len(raw_input_ids) // args.segment_size + 1

    segments_input_ids = []

    for i in range(num_segments - 1):  
        segments_input_ids.append(raw_input_ids[i * args.segment_size: (i + 1) * args.segment_size])

    last_segment_start_index = (num_segments - 1) * args.segment_size
    last_segment_tokens = raw_input_ids[last_segment_start_index:]
    segments_input_ids.append(last_segment_tokens)

    return segments_input_ids

def Semi_Guided_Iterative_Compression(
    segments,
    query,
    model,
    tokenizer,
    args,
    r,
):
    context = "\n".join(segments)
    
    segments = split_segments_input_ids(
        segments, 
        tokenizer, 
        args
    )

    start_ids = tokenizer("<s>", return_tensors="pt").input_ids[0][1:]
    query_input_ids = tokenizer(query, return_tensors="pt").input_ids[0][1:]
    deltas = get_dynamic_compress_ratio(len(segments), args.k_1)
    neg_deltas = [-e for e in get_dynamic_compress_ratio(len(segments), args.k_2)]
    orginal_ratio = 1 - 1.0 * r / get_token_length(context, tokenizer, True)

    prior_input_ids = None
    for idx, segment in enumerate(segments):
        ratio = min(max(deltas[idx] + orginal_ratio, 0),1)
        current_tau_o = min(max(neg_deltas[idx] + args.tau_o, 0),1)

        if ratio == 1:
            if idx > 0:
                prior_input_ids = torch.cat((prior_input_ids, segment), dim=0)
            else:
                prior_input_ids = segment
            continue

        q_input = torch.cat((start_ids, query_input_ids, prior_input_ids, segment), dim=0) \
            if idx > 0 else torch.cat((start_ids, query_input_ids, segment), dim=0)
        q_ppl = get_ppl(
            input_ids=q_input.unsqueeze(0).to(device), 
            granularity="",
            model=model, 
            tokenizer=tokenizer
        )
        q_ppl = q_ppl[len(torch.cat((query_input_ids, prior_input_ids), dim=0)):] if idx > 0 else q_ppl[len(query_input_ids):]

        _input = torch.cat((start_ids, prior_input_ids, segment),dim=0) \
            if idx > 0 else torch.cat((start_ids, segment), dim=0)
        ppl = get_ppl(
            input_ids=_input.unsqueeze(0).to(device), 
            granularity="",
            model=model, 
            tokenizer=tokenizer
        )
        ppl = ppl[len(prior_input_ids):] if idx > 0 else ppl

        contrast_loss = torch.tensor([l - w_l for l, w_l in zip(ppl, q_ppl)])
        threshold = get_estimate_threshold_base_distribution(contrast_loss, ratio * current_tau_o)
        
        need_idx = (contrast_loss > threshold)
        neg_idx = ~need_idx
        res_loss = ppl[neg_idx]
        if len(res_loss) == 0:
            new_need_idx = need_idx
        else:
            ratio2 = 1.0 * (len(segment) * ratio * (1 - current_tau_o)) / len(res_loss)
            threshold2 = get_estimate_threshold_base_distribution(res_loss, ratio2, reverse=args.reverse)
            
            if args.reverse:
                new_need_idx = torch.tensor([~s if (s == False) and (l > threshold2) else s for l, s in zip(ppl, need_idx)])
            else:
                new_need_idx = torch.tensor([~s if (s == False) and (l < threshold2) else s for l, s in zip(ppl, need_idx)])        

        if idx > 0:
            prior_input_ids = torch.cat((prior_input_ids, segment[new_need_idx]), dim=0)
        else:
            prior_input_ids = segment[new_need_idx]

    return tokenizer.decode(prior_input_ids, skip_special_tokens=True)

def get_rank_results(
    context: list,
    question: str,
    rank_method: str,
    condition_in_question: str,
    context_tokens_length: list,
    instruction=None, 
    query_dict=None, 
    bert=None, 
    bert_tokenizer=None,
    model=None,
    tokenizer=None
):
    def get_distance_bm25(corpus, query):
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        idx = [(ii, 0) for ii in (-doc_scores).argsort()]
        return idx

    def get_distance_gzip(corpus, query):
        def get_score(x, y):
            cx, cy = len(gzip.compress(x.encode())), len(gzip.compress(y.encode()))
            cxy = len(gzip.compress(f"{x} {y}".encode()))
            return (cxy - min(cx, cy)) / max(cx, cy)

        import gzip

        doc_scores = [get_score(doc, query) for doc in corpus]
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx

    def get_distance_sentbert(corpus, query):
        from sentence_transformers import SentenceTransformer, util

        # if self.retrieval_model is None or self.retrieval_model_name != rank_method:
        #     self.retrieval_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        #     self.retrieval_model_name = rank_method
        retrieval_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        doc_embeds = retrieval_model.encode(corpus)
        query = retrieval_model.encode(query)
        doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx

    def get_distance_openai(corpus, query):
        import openai
        from sentence_transformers import util

        # openai.api_key = self.open_api_config.get("api_key", "")
        # openai.api_base = self.open_api_config.get(
        #     "api_base", "https://api.openai.com/v1"
        # )
        # openai.api_type = self.open_api_config.get("api_type", "open_ai")
        # openai.api_version = self.open_api_config.get("api_version", "2023-05-15")
        # engine = self.open_api_config.get("engine", "text-embedding-ada-002")

        # def get_embed(text):
        #     return openai.Embedding.create(
        #         input=[text.replace("\n", " ")], engine=engine
        #     )["data"][0]["embedding"]
        api_key = ""
        client = OpenAI(
            base_url="https://api.xty.app/v1", 
            api_key=api_key,
            http_client=httpx.Client(
                base_url="https://api.xty.app/v1",
                follow_redirects=True,
            ),
        )

        def get_embedding(text, model="text-embedding-3-large"):
            text = text.replace("\n", " ")
            return client.embeddings.create(input = [text], model=model).data[0].embedding

        doc_embeds = [get_embedding(i) for i in corpus]
        query = get_embedding(query)
        doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx

    def get_distance_sentbert_bge(corpus, query):
        from sentence_transformers import SentenceTransformer, util

        # if self.retrieval_model is None or self.retrieval_model_name != rank_method:
        #     self.retrieval_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        #     self.retrieval_model_name = rank_method
        retrieval_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        doc_embeds = retrieval_model.encode(
            [i for i in corpus], normalize_embeddings=True
        )
        query = retrieval_model.encode(query, normalize_embeddings=True)
        doc_scores = -util.dot_score(doc_embeds, query).cpu().numpy().reshape(-1)
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx

    def get_distance_bge_ranker(corpus, query):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        pairs = [[i, query] for i in corpus]
        # if self.retrieval_model is None or self.retrieval_model_name != rank_method:
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-large"
            )
            .eval()
            .to(device)
        )
            # self.retrieval_model = [tokenizer, model]
            # self.retrieval_model_name = rank_method
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            scores = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
        idx = [(ii, 0) for ii in np.argsort(-scores.cpu())]
        return idx

    def get_distance_bge_llmembedder(corpus, query):
        from transformers import AutoModel, AutoTokenizer

        # if self.retrieval_model is None or self.retrieval_model_name != rank_method:
        tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
        model = (
            AutoModel.from_pretrained("BAAI/llm-embedder")
            .eval()
            .to(device)
        )
            # self.retrieval_model = [tokenizer, model]
            # self.retrieval_model_name = rank_method

        instruction_qa_query = (
            "Represent this query for retrieving relevant documents: "
        )
        instruction_qa_key = "Represent this document for retrieval: "
        queries = [instruction_qa_query + query for _ in corpus]
        keys = [instruction_qa_key + key for key in corpus]
        with torch.no_grad():
            query_inputs = tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            key_inputs = tokenizer(
                keys,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            query_outputs = model(**query_inputs)
            key_outputs = model(**key_inputs)
            # CLS pooling
            query_embeddings = query_outputs.last_hidden_state[:, 0]
            key_embeddings = key_outputs.last_hidden_state[:, 0]
            # Normalize
            query_embeddings = torch.nn.functional.normalize(
                query_embeddings, p=2, dim=1
            )
            key_embeddings = torch.nn.functional.normalize(
                key_embeddings, p=2, dim=1
            )
            similarity = query_embeddings @ key_embeddings.T
        idx = [(ii, 0) for ii in np.argsort(-similarity[0].cpu())]
        return idx

    def get_distance_longllmlingua(corpus, query):
        # print(corpus)
        context_ppl = [
            get_condition_ppl(
                text=d,
                question=t.format(question=query)
                + " We can get the answer to this question in the given documents.",
                condition_in_question=condition_in_question,
                model=model,
                tokenizer=tokenizer
            ).item()
            for d, dl in zip(corpus, context_tokens_length)
        ]
        sort_direct = -1 if condition_in_question == "none" else 1
        ys = sorted(enumerate(context_ppl), key=lambda x: sort_direct * x[1])
        return ys

    def get_coefficients(source_query, querys, model, tokenizer):

        all_queries = [source_query] + querys
        
        all_tokens = tokenizer(all_queries, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            all_embeddings = model(**all_tokens).last_hidden_state
            all_embeddings = all_embeddings.mean(dim=1)  # 对每个查询的嵌入向量取平均

        source_query_embedding = all_embeddings[0].unsqueeze(0)
        query_embeddings = all_embeddings[1:]
        similarities = torch.nn.functional.cosine_similarity(source_query_embedding, query_embeddings, dim=1)
        similarities = similarities.tolist()

        return similarities   

    def extract_question(input_string):
        start_index = 0
        for i, char in enumerate(input_string):
            if char.isalpha():
                start_index = i
                break    

        return input_string[start_index:]    

    def get_distance_perception(
            corpus, 
            query,
            ins, 
            query_dict, 
            bert, 
            bert_tokenizer
        ):
        querys = [extract_question(e) for e in query_dict[query][:3]]
        # print(querys[0])
        querys.append(normalize_question(query))
        coeffcients = get_coefficients(
            query, 
            querys, 
            bert, 
            bert_tokenizer
        )

        context_ppl = []
        for d in corpus:
            s = 0
            for q, c in zip(querys, coeffcients):
                e = c * get_condition_ppl(
                    text=d,
                    question=ins
                    + t.format(question=q)
                    + " We can get the answer to this question in the given documents.",
                    condition_in_question=condition_in_question,
                    model=model,
                    tokenizer=tokenizer
                ).item()
                s += e
            context_ppl.append(s)

        sort_direct = -1 if condition_in_question == "none" else 1
        ys = sorted(enumerate(context_ppl), key=lambda x: sort_direct * x[1])
        return ys

    method = None
    if rank_method == "bm25":
        method = get_distance_bm25
    elif rank_method == "gzip":
        method = get_distance_gzip
    elif rank_method == "sentbert":
        method = get_distance_sentbert
    elif rank_method == "openai":
        method = get_distance_openai
    elif rank_method in ["longllmlingua", "llmlingua"]:
        method = get_distance_longllmlingua
    elif rank_method == "bge":
        method = get_distance_sentbert_bge
    elif rank_method == "bge_reranker":
        method = get_distance_bge_ranker
    elif rank_method == "bge_llmembedder":
        method = get_distance_bge_llmembedder
    elif rank_method == "perception_compressor":
        method = get_distance_perception
        return method(context, question, instruction, query_dict, bert, bert_tokenizer)
    
    return method(context, question)

def get_template(path):
    with open(path) as f:
        prompt_template = f.read().rstrip("\n")
    return prompt_template

def budget_control(
        instruction, 
        docs, 
        query, 
        ratio,
        tokenizer,
        expand=1,
    ):

    org_prompt = instruction + "\n\n" + "\n".join(docs) + "\n\nQuestion:" + query + "\nAnswer:"
    org_len = get_token_length(org_prompt, tokenizer, False)
    # print(org_len)
    # exit(0)
    retain_tokens = (1.0 * org_len / ratio) * expand
    # print(retain_tokens)

    c = org_prompt.split("\n\n")
    ins_len, q_len = get_token_length(c[0], tokenizer, False), get_token_length(c[-1], tokenizer, False)
    retain_tokens -= (ins_len + q_len)
    target_token = retain_tokens

    r = 0
    retain_docs = []
    for idx, doc in enumerate(docs):
        if idx != (len(docs) - 1):
            retain_tokens -= (get_token_length(doc, tokenizer, False) + 2)
        else:
            retain_tokens -= get_token_length(doc, tokenizer, False)
        retain_docs.append(doc)

        if retain_tokens < 0:
            res = retain_tokens
            break
    
    res = (1.0 * org_len / ratio) * (expand - 1) - res
    
    return retain_docs, res, target_token

def num_tokens_from_string(string, cnt_tokenizer):
    num_tokens = len(cnt_tokenizer(string, return_tensors="pt").input_ids[0]) - 1
    return num_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--compression_rate", type=float)
    parser.add_argument("--segment_size", type=int)
    parser.add_argument("--k_1", type=float)
    parser.add_argument("--tau_o", type=float)
    parser.add_argument("--k_2", type=float)
    parser.add_argument("--coarse_control_coefficient", type=float, default=1)
    parser.add_argument('--reverse', action='store_true')

    parser.add_argument("--save_folder_path", type=str)
    parser.add_argument("--cnt_tokenizer_path", type=str)
    parser.add_argument("--llama_path", type=str)
    parser.add_argument("--bert_path", type=str)
    parser.add_argument("--query_dict_path", type=str)
    parser.add_argument("--qa_data_path", type=str)

    args = parser.parse_args()

    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = "[PAD]"

    model = LlamaForCausalLM.from_pretrained(args.llama_path, torch_dtype=torch.float16).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(args.llama_path)
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    bert = AutoModel.from_pretrained(args.bert_path, torch_dtype=torch.float16).to(device)
    cnt_tokenizer = AutoTokenizer.from_pretrained(args.cnt_tokenizer_path, use_fast=False)
        
    query_dict = load_json(args.query_dict_path)
    dataset = load_jsonl(args.qa_data_path)

    instruction = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."

    if args.save_folder_path:
        if not os.path.exists(args.save_folder_path):
            os.mkdir(args.save_folder_path)


    save_name = f"test.jsonl"
    save_path = os.path.join(args.save_folder_path, save_name)

    for ii, ex in enumerate(tqdm(dataset)):
        
        query = ex["question"]
        
        org_docs = [(f"Document [{idx + 1}](Title:" + e["title"] + ")" + e["text"]) for idx, e in enumerate(ex["ctxs"])]

        idx = get_rank_results(
            org_docs,
            query,
            rank_method="perception_compressor",
            condition_in_question="after",
            context_tokens_length=[0] * len(org_docs),
            instruction=instruction, 
            query_dict=query_dict,
            bert=bert, 
            bert_tokenizer=bert_tokenizer,
            model=model,
            tokenizer=tokenizer 
        ) 

        rerank_docs = [(f"Document [{e[0]}](Title:" + ex["ctxs"][e[0]]["title"] + ")" + ex["ctxs"][e[0]]["text"]) for e in idx]
        
        retain_docs, r, target_token = budget_control(
            instruction, 
            rerank_docs, 
            query, 
            args.compression_rate,
            tokenizer,
            args.coarse_control_coefficient
        )

        compressed_context = Semi_Guided_Iterative_Compression(
            retain_docs,
            query,
            model,
            tokenizer,
            args,
            r=r
        )
        
        compressed_prompt = instruction + "\n\n" + compressed_context + "\n\nQuestion:" + query + "\nAnswer:"
        compressed_tokens = num_tokens_from_string(compressed_prompt, cnt_tokenizer)
        
        org_prompt = instruction + "\n\n" + "\n".join(org_docs) + "\n\nQuestion:" + query + "\nAnswer:"
        org_tokens = num_tokens_from_string(org_prompt, cnt_tokenizer)
        
        d = {
            "compressed_prompt" : compressed_prompt,
            "org_prompt": org_prompt,
            "org_tokens": org_tokens,
            "compressed_tokens": compressed_tokens,
            "actual_compression_rate": 1.0 * org_tokens / compressed_tokens
        }
        
        with open(save_path, "a+") as jf:
            json.dump(d, jf)
            jf.write('\n')  
        
        # exit(0) 