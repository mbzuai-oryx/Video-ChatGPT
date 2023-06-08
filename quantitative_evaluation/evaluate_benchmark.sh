#!/bin/bash

# Define common arguments for all scripts
PRED_DIR="<your_pred_path>"
OUTPUT_DIR="<your_output_dir>"
API_KEY="<your_openai_api_key>"
NUM_TASKS="<number_of_tasks>"

# Run the "correctness" evaluation script
python evaluate_benchmark_1_correctness.py \
  --pred_path "${PRED_DIR}/correctness_pred" \
  --output_dir "${OUTPUT_DIR}/correctness_eval" \
  --output_json "${OUTPUT_DIR}/correctness_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "detailed orientation" evaluation script
python evaluate_benchmark_2_detailed_orientation.py \
  --pred_path "${PRED_DIR}/detailed_orientation_pred" \
  --output_dir "${OUTPUT_DIR}/detailed_eval" \
  --output_json "${OUTPUT_DIR}/detailed_orientation_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "contextual understanding" evaluation script
python evaluate_benchmark_3_context.py \
  --pred_path "${PRED_DIR}/contextual_pred" \
  --output_dir "${OUTPUT_DIR}/context_eval" \
  --output_json "${OUTPUT_DIR}/contextual_understanding_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "temporal understanding" evaluation script
python evaluate_benchmark_4_temporal.py \
  --pred_path "${PRED_DIR}/temporal_understanding_pred" \
  --output_dir "${OUTPUT_DIR}/temporal_eval" \
  --output_json "${OUTPUT_DIR}/temporal_understanding_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "consistency" evaluation script
python evaluate_benchmark_5_consistency.py \
  --pred_path "${PRED_DIR}/consistency_pred" \
  --output_dir "${OUTPUT_DIR}/consistency_eval" \
  --output_json "${OUTPUT_DIR}/consistency_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS


echo "All evaluations completed!"
