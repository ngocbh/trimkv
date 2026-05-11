import argparse
import datetime
import importlib
import json
import os
import sys
import traceback
import warnings
from functools import partial
from dataclasses import dataclass, field, fields
from dotenv import load_dotenv

import numpy as np
import torch
import yaml
import logging
warnings.simplefilter("ignore", category=DeprecationWarning)

import hashlib
from pathlib import Path
from typing import Union

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

from lmms_eval import evaluator, utils
from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.evaluator import request_caching_arg_to_dict
from lmms_eval.loggers import EvaluationTracker, WandbLogger
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import (
    handle_non_serializable,
    make_table,
    simple_parse_args_string,
)
from dataclasses import dataclass, field
from load_model import load_model 

@dataclass
class CompressionConfig:
    method: str = field(default="trimkv", metadata={"help": "Compression method to use"})
    model_path: str = field(default="qwen3_vl", metadata={"help": "Model name"})
    model_args: str = field(default="", metadata={"help": "For compatibility with cli args"})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Attention implementation to use"})
    max_model_len: int = field(default=131072, metadata={"help": "Maximum model length"})
    batch_size: Union[int, str] = field(default=1, metadata={"help": "Batch size for evaluation"})
    max_batch_size: Union[int, str] = field(default=None, metadata={"help": "Max batch size for evaluation"})

    min_pixels: int = field(default=32 * 28 * 28, metadata={"help": "Minimum pixels for image processor"})
    max_pixels: int = field(default=2048 * 28 * 28, metadata={"help": "Maximum pixels for image processor"})
    video_min_frames: int = field(default=4, metadata={"help": "Minimum frames for video processor"})
    video_max_frames: int = field(default=8, metadata={"help": "Maximum frames for video processor"})
    video_min_pixels: int = field(default=4 * 32 * 28 * 28, metadata={"help": "Minimum pixels per frame for video processor"})
    video_max_pixels: int = field(default=8 * 2048 * 28 * 28, metadata={"help": "Maximum pixels per frame for video processor"})
    video_fps: int = field(default=2, metadata={"help": "FPS for video processor"})

    # for KV compression methods
    kv_budget: int = field(default=None, metadata={"help": "KV budget for compression"})

    # for visual token pruning methods (CD PRUNER)
    visual_token_budget: int = field(default=None, metadata={"help": "Visual token budget for compression"})
    textual_extension: bool = field(default=False, metadata={"help": "Whether to extend textual tokens for compression"})
    # For PACT
    pact_config_path: str = field(default="baselines/pact/configs", metadata={"help": "Path to PACT configuration file"})
    pact_method: str = field(default="pact", metadata={"help": "PACT baseline model to use"})
    
    # for trimkv method
    download_from: str = field(default="wandb", metadata={"help": "Where to download the model from"})
    buffer_size: int = field(default=128, metadata={"help": "Buffer size for compression"})
    fixed_kv_budget: bool = field(default=True, metadata={"help": "Set to False for a fair comparison with visual token prunning methods. If set to False, the actual KV budget will be determined dynamically based on the text length, which is num_text_tokens + kv_budget."})
    strategy: str = field(default="fixed_budget", metadata={"help": "Compression strategy to use, [fixed_budget, threshold]"})
    alpha_threshold: float = field(default=0.8, metadata={"help": "Alpha threshold for compression when strategy is set to threshold"})
    lookahead_steps: int = field(default=2, metadata={"help": "Number of lookahead steps for scoring tokens in trimkv"})
    visualization: bool = field(default=False, metadata={"help": "Whether to visualize the compression results"})

    # for RKV compression
    window_size: int = field(default=8, metadata={"help": "Window size for compression"})
    mix_lambda: float = field(default=0.1, metadata={"help": "Mix lambda for compression"})
    retain_ratio: float = field(default=0.2, metadata={"help": "Retain ratio for compression"})
    retain_direction: str = field(default="last", metadata={"help": "Retain direction for compression"})
    divide_method: str = field(default="step_length", metadata={"help": "Method to divide input"})
    divide_length: int = field(default=128, metadata={"help": "Length to divide input"})
    compression_content: str = field(default="all", metadata={"help": "Content to compress"})

    # for streamingllm
    first_tokens: int = field(default=4, metadata={"help": "First tokens for compression"})

    def update_from_dict(self, args):
        for f in fields(self):
            if f.name in args:
                v = args[f.name]
                if v is not None:
                    setattr(self, f.name, v)
        return self

    @property
    def model_type(self):

        model_path = self.model_path.lower().replace("-", "_").replace(" ", "_").replace(".", "_")
        if 'qwen3' in model_path:
            return 'qwen3_vl'
        elif 'llava_1_5' in model_path:
            return 'llava_hf'
        elif 'qwen2_5' in model_path:
            return 'qwen2_5_vl'
        else:
            raise ValueError(f"Unknown base model in path: {model_path}")



def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument("--start_doc_id", type=int, default=0, help="Starting index for evaluation, this is useful for distributed evaluation")
    parser.add_argument(
        "--is_debug",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="Whether to run in debug mode (true/false)"
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lmms-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Compression method to use, e.g. `trimkv`, `dbtrimkv`, `cdpruner`",
    )
    parser.add_argument(
        "--compress_args",
        default="",
        help="String arguments for model, e.g. `compress_memory=True,compress_strategy=alpha,kv_budget=32,max_model_len=32768`",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="Whether to rerun evaluation even if cached results are available",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Run name to be used for logging and saving results",
    )
    parser.add_argument(
        "--launcher_args",
        default=None,
        help="String arguments for launcher for local llm as judge, e.g. `tp=8`, if None then no launcher will be used.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="DEPRECATED: This flag is deprecated and will be removed in a future version. "
        "For debugging, use --log_samples to save all outputs to files. "
        "This flag prints prompts for the first few documents to console, impacting performance.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases",
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="model_outputs",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="If True, applies the chat template to the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument("--process_with_media", action="store_true", help="Whether you will process you dataset with audio, image. By default set to False" "In case some benchmarks need to be processed with media, set this flag to True.")
    parser.add_argument("--force_simple", action="store_true", help="Force the evaluation to use the simple mode of the models")
    args = parser.parse_args()
    return args


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    default_args = parse_eval_args()

    if args is None and len(sys.argv) == 1:
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.                          │")
        print("│ `lmms-eval --model llava --model_path liuhaotian/llava-v1.6-7b --tasks okvqa` │")
        print("│ Use `lmms-eval --help` for more information.                                  │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    # If args were provided, override the defaults
    if args:
        for key, value in vars(args).items():
            setattr(default_args, key, value)

    args = default_args

    if args.is_debug:
        try: 
            import debugpy
        except ImportError:
            raise ImportError("debugpy is not installed. Please install it with `pip install debugpy` to use the debug mode.")
        debugpy.listen(("0.0.0.0", 5675))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached!")
        

    if args.wandb_args:
        if "name" not in args.wandb_args:
            name = f"{args.model}_{args.model_args}_{utils.get_datetime_str(timezone=args.timezone)}"
            name = utils.sanitize_long_string(name)
            args.wandb_args += f",name={name}"
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    # reset logger
    eval_logger.remove()
    # Configure logger with detailed format including file path, function name, and line number
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity, format=log_format)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["VERBOSITY"] = args.verbosity

    args_list = []
    results_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # initialize Accelerator only if not already in a distributed context
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        accelerator = None
        is_main_process = torch.distributed.get_rank() == 0
    else:
        kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
        accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
        if accelerator.is_main_process:
            is_main_process = True
        else:
            is_main_process = False

    for args in args_list:
        try:
            # if is_main_process and args.wandb_args:  # thoughtfully we should only init wandb once, instead of multiple ranks to avoid network traffics and unwanted behaviors.
            #     wandb_logger = WandbLogger()
            results, samples = cli_evaluate_single(args)
            results_list.append(results)

            if accelerator:
                accelerator.wait_for_everyone()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if is_main_process and args.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if args.wandb_log_samples and samples is not None:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:
                    eval_logger.info(f"Logging to Weights and Biases failed due to {e}")
                # wandb_logger.finish()

        except Exception as e:
            if args.verbosity == "DEBUG":
                raise e
            else:
                traceback.print_exc()
                eval_logger.error(f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")
                results_list.append(None)

    for args, results in zip(args_list, results_list):
        # cli_evaluate will return none if the process is not the main process (rank 0)
        if results is not None:
            print(f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, " f"batch_size: {args.batch_size}")
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))

    if args.wandb_args:
        wandb_logger.run.finish()



def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    selected_task_list = args.tasks.split(",") if args.tasks else None

    # Tailored for KV Compression methods
    compression_config = CompressionConfig(
        method=args.method,
        model_path=args.model,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        model_args=args.model_args,
    )
    compression_config_dict = simple_parse_args_string(args.compress_args)
    compression_config.update_from_dict(compression_config_dict)

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=compression_config.model_type)

    # update the evaluation tracker args with the output path and the HF token
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"
    if args.run_name:
        args.hf_hub_log_args += f",run_name={args.run_name}"

    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    eval_logger.info(f"Evaluation tracker args: {evaluation_tracker_args}")

    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if args.write_out:
        eval_logger.warning(
            "DEPRECATION WARNING: --write_out is deprecated and will be removed in v0.5.0. "
            "For debugging and analysis, use --log_samples instead, which saves all model "
            "outputs to files without impacting performance. The --write_out flag only prints "
            "the first few documents to console and provides limited debugging value."
        )

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError("If fewshot_as_multiturn is set, apply_chat_template must be set to True.")

    if (args.num_fewshot is None or args.num_fewshot == 0) and args.fewshot_as_multiturn:
        raise ValueError("If fewshot_as_multiturn is set, num_fewshot must be greater than 0.")

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning("Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub.")

    if args.limit:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if os.environ.get("LMMS_EVAL_PLUGINS", None):
        args.include_path = [args.include_path] if args.include_path else []
        for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
            package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]
            args.include_path.append(package_tasks_location)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format(f"\n - ".join(sorted(task_manager.all_tasks))))
        sys.exit()
    elif args.tasks == "list_groups":
        eval_logger.info(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [task for task in task_list if task not in task_names and "*" not in task]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n" f"{utils.SPACING}Try `lmms-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lmms-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    eval_logger.info(f"Selected Tasks: {task_names}")
    request_caching_args = request_caching_arg_to_dict(cache_requests=args.cache_requests)
    datetime_str = utils.get_datetime_str(timezone=args.timezone)

    print("Running with args:", args)

    model = load_model(compression_config)

    results = evaluator.simple_evaluate(
        model=args.model if model is None else model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        cli_args=args,
        datetime_str=datetime_str,
        distributed_executor_backend="torchrun" if (torch.distributed.is_available() and torch.distributed.is_initialized()) else "accelerate",
        force_simple=args.force_simple,
        launcher_args=args.launcher_args,
        visualization=compression_config.visualization,
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None
        results['compression_config'] = compression_config.__dict__
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(results=results, samples=samples if args.log_samples else None, datetime_str=datetime_str)

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()

        return results, samples
    return None, None


def print_results(args, results):
    print(f"{args.model} ({args.model_args}),\ngen_kwargs: ({args.gen_kwargs}),\nlimit: {args.limit},\nnum_fewshot: {args.num_fewshot},\nbatch_size: {args.batch_size}")
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    load_dotenv()
    cli_evaluate()
