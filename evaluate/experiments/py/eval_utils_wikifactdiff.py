"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

from collections import Counter
import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from baselines.prompt import PromptedModel

from dsets import AttributeSnippets
from util.generate import generate_fast, generate_fast2
from util.perplexity import perplexity

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def apply_edits(model : torch.nn.Module, weights : dict[str, torch.Tensor]) -> dict:
    copy = {}
    for k,v in weights.items():
        if k.startswith('__'):
            k = k[2:]
            copy['__' + k] = getattr(model, k)
            setattr(model, k, v)            
            continue
        param = model.get_parameter(k)
        copy[k] = param.data
        param.data = v
    return copy

def _compute_rewrite_quality_wikifactdiff_functional(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    skip_generation_tests : bool,
    weights_copy
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """
    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = [x['prompt'] for x in record["neighborhood_prompts"]]
    neighborhood_targets_true = [x['expected_object'] for x in record["neighborhood_prompts"]]

    prob_prompts = rewrite_prompts + paraphrase_prompts
    
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct, neighborhood_probs = test_batch_prediction(
        model,
        weights_copy,
        tok,
        prob_prompts,
        target_new['str'],
        target_true['str'],
        neighborhood_prompts,
        neighborhood_targets_true
    )
    rw_p, probs = probs[0], probs[1:]
    rewrite_prompts_probs_ret = [rw_p]
    rewrite_prompts_correct_ret = [targets_correct[0]]

    paraphrase_prompts_probs_ret = probs[:len(prob_prompts)-1]
    paraphrase_prompts_correct_ret = targets_correct[1:]


    
    # Structure the restuls as a dictionary.
    ret = {
        'rewrite_prompts_probs' : rewrite_prompts_probs_ret,
        'paraphrase_prompts_probs' : paraphrase_prompts_probs_ret,
        'neighborhood_prompts_probs' : neighborhood_probs,
        'rewrite_prompts_correct' : rewrite_prompts_correct_ret,
        'paraphrase_prompts_correct' : paraphrase_prompts_correct_ret
    }

    
    if not skip_generation_tests:
        gen_texts = generate_fast2(
            model,
            tok,
            paraphrase_prompts,
            max_out_len=100,
        )
        ngram_entropy = n_gram_entropy(gen_texts)
        ret['ngram_entropy'] = ngram_entropy
    return ret

def get_list(l : list, idx : int, default):
    try:
        return l[idx]
    except IndexError:
        return default

def topk_prob(model, tokenizer, prompt : str, answers : list[str]) -> tuple[list[bool], list[float]]:
    """Returns success flags given a prompt and a set of expected answers, as well as the posterior log probability of the answers. A success flag has a value if True
    the corresponding answer is one of the top-|answers| continuations of the model

    Args:
        model: A causal LM (from huggingface's transformers library)
        tokenizer: Tokenizer (from huggingface)
        prompt (str): A prompt to test (e.g. "The capital of France is")
        answers (list[str]): A list of expected answers (e.g Spain, Italy, ...)

    Returns:
        list[bool], list[float]: Success for each answer, log probability of each answer
    """
    assert all(len(x) for x in answers)
    assert len(set(answers)) == len(answers)
    assert len(prompt)
    save_state = model.training
    model.eval()
    with torch.no_grad():
        prompt = prompt.rstrip(' ')
        expected_tokens = []
        prompt_tokens = tokenizer.encode([prompt]*len(answers), return_tensors="pt").cuda()
        len_prompt = prompt_tokens.shape[-1]
        for answer in answers:
            expected_tokens.append(tokenizer.encode(prompt + ' ' + answer)[len_prompt:])
        is_success = [True]*len(answers)
        prob_answers = [0]*len(answers)
        max_length = max(len(x) for x in expected_tokens)
        offset = 0
        while True:
            probs : torch.Tensor = torch.log_softmax(model(prompt_tokens).logits[:, -1])
            _, top_tokens = probs.topk(len(answers), dim=-1)
            for i, ans_tokens in enumerate(expected_tokens):
                if offset < len(ans_tokens):
                    if ans_tokens[offset] not in top_tokens[i]:
                        is_success[i] = False
                    prob_answers[i] += probs[i,ans_tokens[offset]].item()
                    

            # if all(x is not None for x in is_success):
            #     break
            if offset == max_length-1:
                break
            tokens_cat = torch.tensor([get_list(x,offset+1,0) for x in expected_tokens], dtype=torch.int64, device='cuda')
            prompt_tokens = torch.cat([prompt_tokens, tokens_cat], dim=1)
            offset += 1

    # Recover training state
    if save_state:
        model.train()
    return is_success, prob_answers
        

from scipy import stats
def specificity_test(model, tokenizer, prompts : list[str], weights_copy) -> tuple[list[float], list[int]]:
    """Compute the KL divergence between pre-edit and post-edit next token probability distribution (only for first token). 
    In addition, this function computes the rank of the pre-edit top token after the edit.

    Args:
        model: Transformer model (huggingface)
        tokenizer: Tokenizer (huggingface)
        prompts (list[str]): The prompts to test
        weights_copy (_type_): A copy of the weights that have changed during the edit

    Returns:
        list[float], list[int]: KL divergence of each prompt, ranking for each prompt
    """
    with torch.no_grad():
        prompts_tokens = tokenizer.encode(prompts, return_tensor='pt').cuda()
        copy = apply_edits(model, weights_copy)
        pre_logprobs = torch.log_softmax(model(prompts_tokens).logits, dim=-1)
        apply_edits(model, copy)
        post_logprobs = torch.log_softmax(model(prompts_tokens).logits, dim=-1)
        kl_div = torch.nn.functional.kl_div(pre_logprobs, post_logprobs, log_target=True, reduction='none').sum(-1)

        pre_top_token = pre_logprobs.argmax(dim=-1)
        rank_top_token = []
        
        for i, t in enumerate(pre_top_token):
            rank_top_token.append(stats.percentileofscore(post_logprobs[i], post_logprobs[i,t], kind='rank'))
    return list(kl_div.numpy()), rank_top_token
    

def _compute_rewrite_quality_wikifactdiff_full(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    efficacy_prompt : str,
    generalization_prompts : list[str],
    neighbor_specificity_prompts : list[str],
    random_specificity_prompts : list[str],
    good_answers : list[str],
    bad_answers : list[str],
    weights_copy
) -> typing.Dict:
    """Compute the update quality on a given update instance

    Args:
        model (AutoModelForCausalLM): A huggingface transformer causal language model
        tok (AutoTokenizer): Tokenizer (huggingface)
        efficacy_prompt (str): The prompt used to edit the model
        generalization_prompts (list[str]): Paraphrases of efficacy_prompt
        neighbor_specificity_prompts (list[str]): Prompts of neighbor facts to the edited one
        random_specificity_prompts (list[str]): Prompts of random facts
        good_answers (list[str]): list of good answers to the efficacy_prompt (hence generalization_prompts)
        bad_answers (list[str]): list of bad answers to the efficacy_prompt (hence generalization_prompts)
        weights_copy (_type_): A copy of the weights that have changed during the update

    Returns:
        typing.Dict: A dictionary of metrics describing the quality of the update
    """
    all_answers = good_answers + bad_answers
    n_good = len(good_answers)
    efficacy_success, efficacy_prob = topk_prob(model, tok, efficacy_prompt, all_answers)
    generalization_res = [topk_prob(model, tok, prompt, all_answers) for prompt in generalization_prompts]
    generalization_success, generalization_prob = list(zip(*generalization_res))

    random_kl_div, random_ranking = specificity_test(model, tok, random_specificity_prompts, weights_copy)
    neighbor_kl_div, neighbor_ranking = specificity_test(model, tok, neighbor_specificity_prompts, weights_copy)

    ret = {
        'efficacy' : {
            'good_acc' : efficacy_success[:n_good],
            'good_prob' : efficacy_prob[:n_good],
            'bad_acc' : efficacy_success[n_good:],
            'bad_prob' : efficacy_prob[:n_good]
        },
        'generalization' : {
            'good_acc' : [x[:n_good] for x in generalization_success],
            'bad_acc' : [x[n_good:] for x in generalization_success],
            'good_prob' : [x[:n_good] for x in generalization_prob],
            'bad_prob' : [x[n_good:] for x in generalization_prob],
        },
        'specificity':{
            'random_kl_div' : random_kl_div,
            'random_ranking' : random_ranking,
            'neighbor_kl_div' : neighbor_kl_div,
            'neighbor_ranking' : neighbor_ranking
        }
    }

    
    gen_texts = generate_fast2(
        model,
        tok,
        generalization_prompts,
        max_out_len=100,
    )
    ngram_entropy = n_gram_entropy(gen_texts)
    ret['fluency'] = ngram_entropy
    return ret


def test_batch_prediction(
    model,
    weights_copy,
    tok,
    prefixes: list[str],
    target_new : str,
    target_true : str,
    neighborhood_prefixes : list[str],
    neighborhood_targets_true : list[str]
):
    """
    which_correct: Which target to consider correct for each prefix. Take values from 1 to len(target_list).
    """
    get_length = (lambda x : len(tok.encode(x))) if not isinstance(model, PromptedModel) else (lambda x : model.get_length(x))
    which_correct = [0]*len(prefixes)
    prefix_lens = [get_length(x) for x in prefixes]
    if len(neighborhood_prefixes):
        prefix_neighborhood_lens = [get_length(x) for x in neighborhood_prefixes]
        real_prefix_neighborhood_lens = [len(tok.encode(x)) for x in neighborhood_prefixes]
    
    prompt_tok = tok([
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt"
    ).to('cuda')
    if len(neighborhood_prefixes):
        neigh_prompt_tok = tok(
            [
                f"{prefix} {suffix}"
                for prefix, suffix, in zip(neighborhood_prefixes, neighborhood_targets_true)
            ],
            padding=True,
            return_tensors="pt",
        ).to("cuda")


    with torch.no_grad():
        logits = model(**prompt_tok).logits
    if len(neighborhood_prefixes):
        with torch.no_grad():
            neighborhood_logits = model(**neigh_prompt_tok).logits
            copy = apply_edits(model, weights_copy)
            neighborhood_old_logits = model(**neigh_prompt_tok).logits
            apply_edits(model, copy)
    # if len(neighborhood_prefixes):
    #     logits, neighborhood_logits = logits[:-len(neighborhood_prefixes)], logits[-len(neighborhood_prefixes):]

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)
    if len(neighborhood_prefixes):
        neighborhood_probs = np.zeros((neighborhood_logits.size(0),), dtype=np.float32)
        neighborhood_old_probs = np.zeros((neighborhood_old_logits.size(0),), dtype=np.float32)
        
        
        tokens = tuple(tok(f" {n}")["input_ids"] for n in neighborhood_targets_true)
        choices_len = tuple(len(n) for n in tokens)
        
        for i in range(neighborhood_logits.size(0)):
            cur_len = choices_len[i]

            # Compute suffix probabilities
            for j in range(cur_len):
                cur_tok = tokens[i][j]
                neighborhood_probs[i] += -torch.nn.functional.log_softmax(
                    neighborhood_logits[i, prefix_neighborhood_lens[i] + j - 1, :], dim=0
                )[cur_tok].item()
                neighborhood_old_probs[i] += -torch.nn.functional.log_softmax(
                    neighborhood_old_logits[i, real_prefix_neighborhood_lens[i] + j - 1, :], dim=0
                )[cur_tok].item()
            neighborhood_old_probs[i] /= cur_len
            neighborhood_probs[i] /= cur_len
    ret_neighborhood = [
        {"post_object_true": neighborhood_probs[i].item(), "pre_object_true": neighborhood_old_probs[i].item()}
        for i in range(0, len(neighborhood_probs))
    ] if len(neighborhood_prefixes) else []
    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct, ret_neighborhood



def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()
