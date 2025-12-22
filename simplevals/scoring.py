import os
import json
import math
import argparse
import glob
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from llms.utils import get_llm
from .prompts import HUMAN_GUIDELINE


def evaluate(judge, preds, output_path, major_vote_count=5, include_image=False):
    print('include image', include_image)
    thread_id2image = {}
    thread_id2gt = {}
    thread_id2question = {}

    # Load dataset
    for row in load_dataset("VisTai/vistw-dialogue", split="test"):
        tid = row['thread_id']
        thread_id2question[tid] = row['question']
        thread_id2image[tid] = row['image']
        thread_id2gt[tid] = row['ground_truth']

    # Check for existing results and load them
    existing_results = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    existing_results[data['thread_id']] = data
                except:
                    continue
        print(f"Loaded {len(existing_results)} existing results for resuming")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open output file in append mode for continuous writing
    with open(output_path, 'a', encoding='utf-8') as out_file:
        # Process each prediction
        for pred in tqdm(preds, desc="Evaluating responses", dynamic_ncols=True):
            thread_id = pred['question_id']

            # Skip if already processed
            if thread_id in existing_results:
                continue

            question = thread_id2question[thread_id]
            gt = thread_id2gt[thread_id]
            image = thread_id2image[thread_id] if include_image else None

            # Format prompt with question, response, and ground truth
            prompt = HUMAN_GUIDELINE.format(
                question=question,
                response=pred['llm_response'],
                ground_truth=gt
            )

            # Get multiple evaluations from judge
            responses = []
            for _ in range(int(major_vote_count*1.5)):  # Try up to 10 times to get major_vote_count valid responses
                response, stats = judge(prompt, image, temperature=0.7)
                try:
                    score = int(response.split('[評分]: ')[-1])
                    if not math.isnan(score):# don't ask me what happen here, it just happened
                        responses.append({
                            'response': response,
                            'score': score,
                            'stats': stats
                        })
                except ValueError:
                    continue

                if len(responses) >= major_vote_count:
                    break

            # Skip if couldn't get enough valid responses
            if not responses:
                continue

            # Calculate average score
            score_board = {
                'thread_id': thread_id,
                'judge_responses': responses,
                'score': sum([r['score'] for r in responses])/len(responses)
            }

            # Write result to file immediately for auto-resume capability
            out_file.write(json.dumps(score_board, ensure_ascii=False) + '\n')
            out_file.flush()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate LLM responses")
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL file(s) containing question_id and llm_response. Supports wildcards (e.g., 'logs/*.jsonl' or 'results_*.jsonl')")
    parser.add_argument("--series", default="gemini", choices=["gemini_dev", "gemini", "openai"], help="LLM series")
    parser.add_argument("--model", default="gemini-2.0-flash-001", help="Model name")
    parser.add_argument("--include_image", action="store_true", help="Include images in evaluation")
    parser.add_argument("--voting_count", type=int, default=5, help="Number of evaluations per response")
    return parser.parse_args()


def pretty_print_results(output_path):
    """Print a summary of evaluation results with statistics."""
    if not os.path.exists(output_path):
        print("No results file found.")
        return

    results = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                results.append(data)
            except:
                continue

    if not results:
        print("No valid results found.")
        return

    # Calculate statistics
    scores = [r['score'] for r in results]
    avg_score = np.mean(scores)
    median_score = np.median(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    std_dev = np.std(scores)

    # Count scores by range
    score_dist = {
        "0-1": sum(1 for s in scores if 0 <= s <= 1),
        "2-3": sum(1 for s in scores if 2 <= s <= 3),
        "4-5": sum(1 for s in scores if 4 <= s <= 5),
        "6-7": sum(1 for s in scores if 6 <= s <= 7),
        "8-9": sum(1 for s in scores if 8 <= s <= 9),
        "10": sum(1 for s in scores if s == 10)
    }

    # Print results in a nice format
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS SUMMARY")
    print("="*50)
    print(f"Total responses evaluated: {len(results)}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Median score: {median_score:.2f}")
    print(f"Score range: {min_score:.1f} - {max_score:.1f}")
    print(f"Standard deviation: {std_dev:.2f}")
    print("\nScore distribution:")

    # Calculate the maximum count for scaling the histogram
    max_count = max(score_dist.values())
    scale_factor = 40 / max_count if max_count > 0 else 1

    for range_label, count in score_dist.items():
        bar_length = int(count * scale_factor)
        bar = "█" * bar_length
        percentage = (count / len(scores)) * 100 if scores else 0
        print(f"  {range_label:>4}: {count:>4} ({percentage:>5.1f}%) {bar}")

    print("="*50)
    print(f"Results saved to: {output_path}")
    print("="*50 + "\n")


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Find all files matching the wildcard pattern
    input_files = glob.glob(args.input_jsonl)

    if not input_files:
        print(f"Error: No files found matching pattern: {args.input_jsonl}")
        return

    print(f"Found {len(input_files)} file(s) matching pattern: {args.input_jsonl}")
    for f in input_files:
        print(f"  - {f}")
    print()

    # Initialize judge LLM once for all files
    judge = get_llm(args.model, series=args.series)

    # Process each file individually
    for file_idx, input_file in enumerate(input_files, 1):
        print("=" * 70)
        print(f"Processing file {file_idx}/{len(input_files)}: {input_file}")
        print("=" * 70)

        # Load predictions from current file
        preds = []
        pred_thread_ids = set()
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pred = json.loads(line.strip())
                    preds.append(pred)
                    pred_thread_ids.add(pred['question_id'])
                except:
                    continue

        print(f"Loaded {len(preds)} predictions from this file")

        # Create output filename for this specific file
        input_base = os.path.basename(input_file).replace(".jsonl", "")
        output_dir = os.path.join(os.path.dirname(input_file), "score_board")

        model_name = args.model
        if '/' in model_name:
            model_name = model_name.split('/')[-1].split(':')[0]
        output_filename = f"{input_base}_scores_{model_name}_{args.series}_img{int(args.include_image)}_vote{args.voting_count}.jsonl"
        output_path = os.path.join(output_dir, output_filename)

        # Check existing progress by matching thread_ids
        scored_thread_ids = set()
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if 'thread_id' in result:
                            scored_thread_ids.add(result['thread_id'])
                    except:
                        continue

        # Calculate how many predictions from this file are already scored
        matching_scored = pred_thread_ids & scored_thread_ids
        existing_count = len(matching_scored)
        remaining = len(preds) - existing_count

        if remaining == 0:
            print(f"✓ Already complete ({existing_count}/{len(preds)} scored)")
            print(f"  Output: {output_path}")
            print(f"  Skipping...\n")
            continue
        elif existing_count > 0:
            print(f"Resuming: {existing_count}/{len(preds)} already scored, {remaining} remaining")
        else:
            print(f"Starting fresh: 0/{len(preds)} scored")

        print(f"Output will be saved to: {output_path}\n")

        # Run evaluation for this file
        evaluate(
            judge=judge,
            preds=preds,
            output_path=output_path,
            major_vote_count=args.voting_count,
            include_image=args.include_image
        )

        # Print summary of results for this file
        pretty_print_results(output_path)
        print()

    print("=" * 70)
    print(f"Completed processing all {len(input_files)} file(s)")
    print("=" * 70)


if __name__ == "__main__":
    main()