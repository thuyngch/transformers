#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from tqdm import tqdm, trange
import argparse, glob, logging, os, random, timeit

from transformers import (
	WEIGHTS_NAME,
	AdamW,
	AlbertConfig,
	AlbertForQuestionAnswering,
	AlbertTokenizer,
	BertConfig,
	BertForQuestionAnswering,
	BertTokenizer,
	DistilBertConfig,
	DistilBertForQuestionAnswering,
	DistilBertTokenizer,
	RobertaConfig,
	RobertaForQuestionAnswering,
	RobertaTokenizer,
	XLMConfig,
	XLMForQuestionAnswering,
	XLMTokenizer,
	XLNetConfig,
	XLNetForQuestionAnswering,
	XLNetTokenizer,
	get_linear_schedule_with_warmup,
	squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
	compute_predictions_log_probs,
	compute_predictions_logits,
	squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
	(tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, XLNetConfig, XLMConfig)),
	(),
)

MODEL_CLASSES = {
	"bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
	"roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
	"xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
	"xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
	"distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
	"albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
}


#------------------------------------------------------------------------------
#  Utils
#------------------------------------------------------------------------------
def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
	return tensor.detach().cpu().tolist()

def evaluate(args, model, tokenizer, prefix=""):
	dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

	if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(args.output_dir)

	args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(dataset)
	eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

	# multi-gpu evaluate
	if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
		model = torch.nn.DataParallel(model)

	# Eval!
	logger.info("***** Running evaluation {} *****".format(prefix))
	logger.info("  Num examples = %d", len(dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)

	all_results = []
	start_time = timeit.default_timer()

	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		model.eval()
		batch = tuple(t.to(args.device) for t in batch)

		with torch.no_grad():
			inputs = {
				"input_ids": batch[0],
				"attention_mask": batch[1],
				"token_type_ids": batch[2],
			}

			if args.model_type in ["xlm", "roberta", "distilbert"]:
				del inputs["token_type_ids"]

			example_indices = batch[3]

			# XLNet and XLM use more arguments for their predictions
			if args.model_type in ["xlnet", "xlm"]:
				inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

			outputs = model(**inputs)

		for i, example_index in enumerate(example_indices):
			eval_feature = features[example_index.item()]
			unique_id = int(eval_feature.unique_id)

			output = [to_list(output[i]) for output in outputs]

			# Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
			# models only use two.
			if len(output) >= 5:
				start_logits = output[0]
				start_top_index = output[1]
				end_logits = output[2]
				end_top_index = output[3]
				cls_logits = output[4]

				result = SquadResult(
					unique_id,
					start_logits,
					end_logits,
					start_top_index=start_top_index,
					end_top_index=end_top_index,
					cls_logits=cls_logits,
				)

			else:
				start_logits, end_logits = output
				result = SquadResult(unique_id, start_logits, end_logits)

			all_results.append(result)

	evalTime = timeit.default_timer() - start_time
	logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

	# Compute predictions
	output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
	output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

	if args.version_2_with_negative:
		output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
	else:
		output_null_log_odds_file = None

	# XLNet and XLM use a more complex post-processing procedure
	if args.model_type in ["xlnet", "xlm"]:
		start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
		end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

		predictions = compute_predictions_log_probs(
			examples,
			features,
			all_results,
			args.n_best_size,
			args.max_answer_length,
			output_prediction_file,
			output_nbest_file,
			output_null_log_odds_file,
			start_n_top,
			end_n_top,
			args.version_2_with_negative,
			tokenizer,
			args.verbose_logging,
		)
	else:
		predictions = compute_predictions_logits(
			examples,
			features,
			all_results,
			args.n_best_size,
			args.max_answer_length,
			args.do_lower_case,
			output_prediction_file,
			output_nbest_file,
			output_null_log_odds_file,
			args.verbose_logging,
			args.version_2_with_negative,
			args.null_score_diff_threshold,
			tokenizer,
		)

	# Print
	print("\n[Context]\n{}\n".format(examples[0].context_text))
	for idx, (example, prediction) in enumerate(zip(examples, predictions)):
		print("[Sample {}]".format(idx+1))
		print("Question: {}".format(example.question_text))
		print("Answer: {}".format(predictions[prediction]))
		print("Possible answers:")
		for ans in example.answers:
			print("\t{}".format(ans['text']))
		print()

	# Compute the F1 and exact scores.
	results = squad_evaluate(examples, predictions)
	return results

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
	if args.local_rank not in [-1, 0] and not evaluate:
		# Make sure only the first process in distributed training process the dataset, and the others will use the cache
		torch.distributed.barrier()

	# Load data features from cache or dataset file
	input_dir = args.data_dir if args.data_dir else "."
	cached_features_file = os.path.join(
		input_dir,
		"cached_{}_{}_{}".format(
			"dev" if evaluate else "train",
			list(filter(None, args.model_name_or_path.split("/"))).pop(),
			str(args.max_seq_length),
		),
	)

	# Init features and dataset from cache if it exists
	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features_and_dataset = torch.load(cached_features_file)
		features, dataset, examples = (
			features_and_dataset["features"],
			features_and_dataset["dataset"],
			features_and_dataset["examples"],
		)
	else:
		logger.info("Creating features from dataset file at %s", input_dir)

		if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
			try:
				import tensorflow_datasets as tfds
			except ImportError:
				raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

			if args.version_2_with_negative:
				logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

			tfds_examples = tfds.load("squad")
			examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
		else:
			processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
			if evaluate:
				examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
			else:
				examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

		features, dataset = squad_convert_examples_to_features(
			examples=examples,
			tokenizer=tokenizer,
			max_seq_length=args.max_seq_length,
			doc_stride=args.doc_stride,
			max_query_length=args.max_query_length,
			is_training=not evaluate,
			return_dataset="pt",
			threads=args.threads,
		)

		if args.local_rank in [-1, 0]:
			logger.info("Saving features into cached file %s", cached_features_file)
			torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

	if args.local_rank == 0 and not evaluate:
		# Make sure only the first process in distributed training process the dataset, and the others will use the cache
		torch.distributed.barrier()

	if output_examples:
		return dataset, examples, features
	return dataset


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--model_type",
		default=None,
		type=str,
		required=True,
		help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
	)
	parser.add_argument(
		"--model_name_or_path",
		default=None,
		type=str,
		required=True,
		help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		type=str,
		required=True,
		help="The output directory where the model checkpoints and predictions will be written.",
	)

	# Other parameters
	parser.add_argument(
		"--data_dir",
		default=None,
		type=str,
		help="The input data dir. Should contain the .json files for the task."
		+ "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
	)
	parser.add_argument(
		"--train_file",
		default=None,
		type=str,
		help="The input training file. If a data dir is specified, will look for the file there"
		+ "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
	)
	parser.add_argument(
		"--predict_file",
		default=None,
		type=str,
		help="The input evaluation file. If a data dir is specified, will look for the file there"
		+ "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
	)
	parser.add_argument(
		"--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
	)
	parser.add_argument(
		"--tokenizer_name",
		default="",
		type=str,
		help="Pretrained tokenizer name or path if not the same as model_name",
	)
	parser.add_argument(
		"--cache_dir",
		default="",
		type=str,
		help="Where do you want to store the pre-trained models downloaded from s3",
	)

	parser.add_argument(
		"--version_2_with_negative",
		action="store_true",
		help="If true, the SQuAD examples contain some that do not have an answer.",
	)
	parser.add_argument(
		"--null_score_diff_threshold",
		type=float,
		default=0.0,
		help="If null_score - best_non_null is greater than the threshold predict null.",
	)

	parser.add_argument(
		"--max_seq_length",
		default=384,
		type=int,
		help="The maximum total input sequence length after WordPiece tokenization. Sequences "
		"longer than this will be truncated, and sequences shorter than this will be padded.",
	)
	parser.add_argument(
		"--doc_stride",
		default=128,
		type=int,
		help="When splitting up a long document into chunks, how much stride to take between chunks.",
	)
	parser.add_argument(
		"--max_query_length",
		default=64,
		type=int,
		help="The maximum number of tokens for the question. Questions longer than this will "
		"be truncated to this length.",
	)
	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument(
		"--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
	)
	parser.add_argument(
		"--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
	)

	parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument(
		"--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
	)
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument(
		"--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
	)
	parser.add_argument(
		"--max_steps",
		default=-1,
		type=int,
		help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
	)
	parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
	parser.add_argument(
		"--n_best_size",
		default=20,
		type=int,
		help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
	)
	parser.add_argument(
		"--max_answer_length",
		default=30,
		type=int,
		help="The maximum length of an answer that can be generated. This is needed because the start "
		"and end predictions are not conditioned on one another.",
	)
	parser.add_argument(
		"--verbose_logging",
		action="store_true",
		help="If true, all of the warnings related to data processing will be printed. "
		"A number of warnings are expected for a normal SQuAD evaluation.",
	)

	parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
	parser.add_argument(
		"--eval_all_checkpoints",
		action="store_true",
		help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
	)
	parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
	parser.add_argument(
		"--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
	)
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
	)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

	parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
	parser.add_argument(
		"--fp16",
		action="store_true",
		help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
	)
	parser.add_argument(
		"--fp16_opt_level",
		type=str,
		default="O1",
		help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
		"See details at https://nvidia.github.io/apex/amp.html",
	)
	parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
	parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

	parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
	args = parser.parse_args()

	if (
		os.path.exists(args.output_dir)
		and os.listdir(args.output_dir)
		and args.do_train
		and not args.overwrite_output_dir
	):
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				args.output_dir
			)
		)

	# Setup distant debugging if needed
	if args.server_ip and args.server_port:
		# Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
		import ptvsd

		print("Waiting for debugger attach")
		ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
		ptvsd.wait_for_attach()

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend="nccl")
		args.n_gpu = 1
	args.device = device

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		args.local_rank,
		device,
		args.n_gpu,
		bool(args.local_rank != -1),
		args.fp16,
	)

	# Set seed
	set_seed(args)

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		# Make sure only the first process in distributed training will download model & vocab
		torch.distributed.barrier()

	args.model_type = args.model_type.lower()
	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(
		args.config_name if args.config_name else args.model_name_or_path,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)
	tokenizer = tokenizer_class.from_pretrained(
		args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
		do_lower_case=args.do_lower_case,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)
	model = model_class.from_pretrained(
		args.model_name_or_path,
		from_tf=bool(".ckpt" in args.model_name_or_path),
		config=config,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)

	if args.local_rank == 0:
		# Make sure only the first process in distributed training will download model & vocab
		torch.distributed.barrier()

	model.to(args.device)

	logger.info("Training/evaluation parameters %s", args)

	# Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
	# Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
	# remove the need for this code, but it is still valid.
	if args.fp16:
		try:
			import apex

			apex.amp.register_half_function(torch, "einsum")
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

	# Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
	results = {}
	if args.do_eval and args.local_rank in [-1, 0]:
		if args.do_train:
			logger.info("Loading checkpoints saved during training for evaluation")
			checkpoints = [args.output_dir]
			if args.eval_all_checkpoints:
				checkpoints = list(
					os.path.dirname(c)
					for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
				)
				logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
		else:
			logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
			checkpoints = [args.model_name_or_path]

		logger.info("Evaluate the following checkpoints: %s", checkpoints)

		for checkpoint in checkpoints:
			# Reload the model
			global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
			model = model_class.from_pretrained(checkpoint)
			model.to(args.device)

			# Evaluate
			result = evaluate(args, model, tokenizer, prefix=global_step)

			result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
			results.update(result)

	logger.info("Results: {}".format(results))
	return results


if __name__ == "__main__":
	main()
