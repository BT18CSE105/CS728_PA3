import torch
from tqdm import tqdm
from utils import PromptUtils
import random 

def select_retrieval_heads(train_queries, model, tokenizer, tools, device, max_heads=20):
    # TODO 3: Head selection
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.

    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns(query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)

    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """

    import os

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # accumulate scores per head
    head_scores = torch.zeros(num_layers, num_heads, device=device)
    strategy = os.environ.get("HEAD_SELECTION_STRATEGY", "reciprocal_rank").strip().lower()
    valid_strategies = {"reciprocal_rank", "avg_gold_attention", "top5_hits"}
    if strategy not in valid_strategies:
        strategy = "reciprocal_rank"

    for qix in tqdm(range(len(train_queries))):

        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)
        putils = PromptUtils(
        tokenizer=tokenizer, 
        doc_ids=tool_ids, 
        dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions 

        # Add your head scoring logic after this line
        query_prefix = (
            putils.prompt_prefix
            + putils.all_docs_info_string
            + putils.prompt_seperator
            + putils.add_text1
            + putils.prompt_seperator
            + "Query: "
        )
        query_start = len(tokenizer(query_prefix, add_special_tokens=False).input_ids)
        query_length = len(tokenizer(question, add_special_tokens=False).input_ids)
        query_end = query_start + query_length
        gold_tool_id = map_docname_id[gold_tool_name]

        per_head_doc_scores = torch.zeros(num_layers, num_heads, len(item_spans), device=device)
        for layer_id, layer_attn in enumerate(attentions):
            query_attn = layer_attn[0, :, query_start:query_end, :]
            for doc_idx, (doc_start, doc_end) in enumerate(item_spans):
                per_head_doc_scores[layer_id, :, doc_idx] = query_attn[:, :, doc_start:doc_end].mean(dim=(1, 2))

        if strategy == "avg_gold_attention":
            head_scores += per_head_doc_scores[:, :, gold_tool_id]
        else:
            ranked_docs = torch.argsort(per_head_doc_scores, dim=-1, descending=True)
            gold_ranks = (ranked_docs == gold_tool_id).float().argmax(dim=-1).float()
            if strategy == "top5_hits":
                head_scores += (gold_ranks < 5).float()
            else:
                head_scores += 1.0 / (gold_ranks + 1.0)

    # TODO: select top heads
    flat_head_scores = head_scores.reshape(-1)
    top_k = min(max_heads, flat_head_scores.numel())
    top_indices = torch.topk(flat_head_scores, k=top_k).indices.tolist()
    selected_heads = [(idx // num_heads, idx % num_heads) for idx in top_indices]

    # example expected format:
    # [(layer1, head3), (layer5, head10), ...]
    assert len(selected_heads) == top_k
    return selected_heads
