import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.identity import IdentityHyperParams, identity_rewrite
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from baselines.prompt import PromptHyperParams, prompt_rewrite
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    FunctionalWikiFactDiffDataset,
    get_tfidf_vectorizer,
    WikiFactDiffDataset
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_wikifactdiff import compute_rewrite_quality_wikifactdiff_functional
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util import globals
import os.path as osp
from pathlib import Path
# torch.cuda.set_device(1)



ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "IDENTITY" : (IdentityHyperParams, identity_rewrite),
    "PROMPT" : (PromptHyperParams, prompt_rewrite)
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "wfd" : (FunctionalWikiFactDiffDataset, compute_rewrite_quality_wikifactdiff_functional)
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    dataset_path : str = None,
    results_dir : str = None,
    use_random_neighbors : bool = False,
    n_partitions = 1,
    offset = 0
):
    assert offset < n_partitions
    if results_dir is None:
        results_dir = globals.RESULTS_DIR
    else:
        results_dir = Path(results_dir)
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := results_dir / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = results_dir / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = results_dir / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else globals.HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        try:
            shutil.copyfile(params_path, run_dir / "params.json")
        except FileNotFoundError:
            pass
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    # print("Loading dataset, attribute snippets, tf-idf data")
    # snips = AttributeSnippets(globals.DATA_DIR) if not skip_generation_tests else None
    # vec = get_tfidf_vectorizer(globals.DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(globals.DATA_DIR, tok=tok, size=dataset_size_limit, path=dataset_path, use_random_neighbors=use_random_neighbors)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            globals.KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    for chunk_pos, record_chunks in enumerate(chunks(ds, num_edits), start=offset):
        if chunk_pos % n_partitions != 0:
            continue
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        start = time()
        failed = False
        try:
            edited_model, weights_copy = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks
                ],
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
                **etc_args,
            )
        except RuntimeError:
            failed = True
            edited_model = model
            weights_copy = {}
            print('Update FAILED! Skipping..')
        exec_time = time() - start
        print("Execution took", exec_time)

        # Evaluate new model
        start = time()
        for record in record_chunks:
            out_file = Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue

            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    skip_generation_tests,
                    weights_copy
                ),
                "failed" : failed
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                if k.startswith('__'):
                    if alg_name != 'PROMPT':
                        raise Exception('This block should not be executed unless we use the PROMPT algorithm! Fix this!')
                    continue
                nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND", "IDENTITY", "PROMPT"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "wfd"],
        default="wfd",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default="run_000",
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate Dataset to first n records.",
    )
    parser.add_argument(
        "--use_random_neighbors",
        action='store_true',
        help="Use random neighbors instead of TF-IDF neighbors",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=globals.RESULTS_DIR,
        help="Path to the directory that will contain evaluation results. Defaults to %s" % globals.RESULTS_DIR,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the selected dataset (if needed)",
    )
    parser.add_argument(
        "--n_partitions",
        type=int,
        default=1,
        help="If we choose for example (n_partitions=5, offset=2), the updates are performed on the 2nd, 7th, 12th, etc. instances of the dataset only. This is useful to parallelize the updates (n_partitions is the number of splits and offset is the chosen split).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="(See --n_partitions help section)",
    )
    parser.add_argument(
        "--cache_folder",
        type=str,
        default=None,
        help="Where to store the files needed to run update algorithms such as ROME (wikipedia stats), MEND (weights), MEMIT (wikipedia stats). By default, there will be stored with the source code",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()
    if args.cache_folder is not None:
        from util import globals
        globals.CACHE_FOLDER = args.cache_folder
        globals.DATA_DIR = osp.join(args.cache_folder, globals.DATA_DIR)
        globals.STATS_DIR = osp.join(args.cache_folder, globals.STATS_DIR)

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        results_dir=args.results_dir,
        dataset_path=args.dataset_path,
        use_random_neighbors=args.use_random_neighbors,
        n_partitions=args.n_partitions,
        offset=args.offset
    )
