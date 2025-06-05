import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from utils.config_loader import load_config, AppConfig
from utils.logging_setup import setup_logging
from utils.ngram_utils import NgramUtil
from data_handling.database import DatabaseManager
from data_handling.prompt_loader import PromptLoader
from llm_interface.api_client import ApiClient
from llm_interface.chat_formatter import ChatTemplateFormatter
from classification.refusal_detector import RefusalDetector
from core.explorer import BeamExplorer
from core.models import PromptData


logger = logging.getLogger(__name__) # Will be configured by setup_logging

def process_single_prompt(
    prompt_data: PromptData,
    config: AppConfig,
    model_db_id: int,
    # Shared instances
    api_client: ApiClient,
    chat_formatter: Optional[ChatTemplateFormatter],
    refusal_detector: RefusalDetector,
    db_manager: DatabaseManager, # Each thread will get its own connection from this manager
    ngram_util: NgramUtil
):
    """
    Processes a single prompt using BeamExplorer.
    This function is designed to be run in a thread.
    """
    try:
        logger.info(f"Starting exploration for prompt ID: {prompt_data.original_id} (DB ID: {prompt_data.db_id}) from dataset '{prompt_data.source_dataset_name}'") # Added source_dataset_name
        
        explorer = BeamExplorer(
            config=config,
            api_client=api_client, # ApiClient is designed to be thread-safe with shared session
            chat_formatter=chat_formatter,
            refusal_detector=refusal_detector, # RefusalDetector is thread-safe
            db_manager=db_manager, # DatabaseManager handles thread-local connections
            ngram_util=ngram_util,
            model_db_id=model_db_id,
            prompt_data=prompt_data
        )
        # Initial call to explorer
        explorer.explore(current_beam_path_raw=[], current_depth=0, current_path_decoded_words=[])
        
        # Update last processed prompt ID for this specific dataset and model
        db_manager.set_last_processed_prompt_id(
            model_api_name=config.api.model_name,
            dataset_name=prompt_data.source_dataset_name, # Use the actual dataset name
            prompt_id=prompt_data.original_id
        )
        logger.info(f"Finished exploration for prompt ID: {prompt_data.original_id}")
        return prompt_data.original_id, True
    except Exception as e:
        logger.error(f"Error processing prompt ID {prompt_data.original_id}: {e}", exc_info=True)
        return prompt_data.original_id, False
    finally:
        # Important for thread-local DB connections if they are not automatically closed
        # db_manager.close_connection() # Or manage this at the end of the thread's life
        pass


def main():
    parser = argparse.ArgumentParser(description="AntiRefusal Sampler: Explore token paths for refusal analysis.")
    parser.add_argument("--config", type=str, default="configs/current_config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--resume", action="store_true", help="Resume processing from the last saved state for each dataset.")
    # parser.add_argument("--max-prompts-per-dataset", type=int, default=None, help="Maximum number of new prompts to process per dataset in this run.")

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.logging_level)

    logger.info("AntiRefusal Sampler starting...")
    logger.info(f"Using configuration from: {args.config}")
    if args.resume:
        logger.info("Resume mode enabled.")

    db_manager = DatabaseManager(config.database_path)
    
    # Get or create model_db_id
    # For simplicity, storing a subset of config related to the model run
    model_config_details = {
        "api": dict(config.api), # Convert Pydantic-like to dict
        "exploration": dict(config.exploration),
        "sampling_beam_starters": dict(config.sampling_beam_starters),
        "quotas": dict(config.quotas)
    }
    model_db_id = db_manager.add_model_if_not_exists(config.api.model_name, model_config_details)
    logger.info(f"Using model '{config.api.model_name}' (DB ID: {model_db_id})")

    api_client = ApiClient(
        base_url=config.api.base_url,
        api_key=config.api.api_key,
        model_name=config.api.model_name, # This is the model name for the API
        timeout_seconds=config.api.timeout_seconds,
        pool_size=config.max_workers + 5 # Give some buffer for API client connections
    )

    chat_formatter = None
    if config.api.chat_template_model_id:
        chat_formatter = ChatTemplateFormatter(
            model_id=config.api.chat_template_model_id,
            # system_prompt can be added to config if needed, or assumed empty for /completions
        )
        logger.info(f"Chat template formatter initialized for '{config.api.chat_template_model_id}'.")

    refusal_detector = RefusalDetector.get(
        model_id=config.refusal_classifier.model_id,
        # device can be auto-detected or configured
    )
    if refusal_detector._failed:
        logger.error("Failed to initialize refusal detector. Exiting.")
        return

    ngram_util = NgramUtil(language=config.ngrams.language)

    prompts_to_process_all_datasets = []

    for dataset_conf_dict in config.datasets:
        dataset_conf = dict(dataset_conf_dict) # Convert Pydantic-like to dict
        dataset_name = dataset_conf['name']
        logger.info(f"Loading prompts for dataset: {dataset_name}")
        
        last_processed_id_for_this_dataset = None
        if args.resume:
            last_processed_id_for_this_dataset = db_manager.get_last_processed_prompt_id(config.api.model_name, dataset_name)
            if last_processed_id_for_this_dataset:
                logger.info(f"Resuming dataset '{dataset_name}' after prompt ID: {last_processed_id_for_this_dataset}")

        prompt_loader = PromptLoader(dataset_conf)
        
        processed_in_this_run_count = 0
        found_resume_point = not last_processed_id_for_this_dataset # If no resume ID, start from beginning

        for p_data_raw in prompt_loader.load_prompts():
            if not found_resume_point:
                if p_data_raw.original_id == last_processed_id_for_this_dataset:
                    found_resume_point = True
                continue # Skip until resume point is found

            # Add prompt to DB to get its prompt_db_id
            prompt_db_id = db_manager.add_prompt_if_not_exists(
                original_prompt_id=p_data_raw.original_id,
                source_dataset_name=dataset_name, # Pass dataset name
                category=p_data_raw.category,
                prompt_text=p_data_raw.text
            )
            # Augment PromptData with db_id and source_dataset_name
            p_data_for_processing = PromptData(
                original_id=p_data_raw.original_id,
                category=p_data_raw.category,
                text=p_data_raw.text,
                db_id=prompt_db_id,
                source_dataset_name=dataset_name # Store this for logging/state
            )
            prompts_to_process_all_datasets.append(p_data_for_processing)
            processed_in_this_run_count += 1
            
            # if args.max_prompts_per_dataset and processed_in_this_run_count >= args.max_prompts_per_dataset:
            #     logger.info(f"Reached max_prompts_per_dataset ({args.max_prompts_per_dataset}) for '{dataset_name}'.")
            #     break
        
        if last_processed_id_for_this_dataset and not found_resume_point:
            logger.warning(f"Resume ID '{last_processed_id_for_this_dataset}' not found in dataset '{dataset_name}'. Processing all prompts.")


    if not prompts_to_process_all_datasets:
        logger.info("No new prompts to process across all datasets. Exiting.")
        return

    logger.info(f"Total prompts to process across all datasets: {len(prompts_to_process_all_datasets)}")

    # Use ThreadPoolExecutor for parallel processing of prompts
    # Each thread will handle one prompt from start to finish.
    # Shared objects (api_client, chat_formatter, refusal_detector, db_manager, ngram_util)
    # must be thread-safe or manage thread-local resources.
    
    num_workers = config.max_workers
    logger.info(f"Starting ThreadPoolExecutor with {num_workers} workers.")
    
    successful_prompts = 0
    failed_prompts = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_single_prompt,
                prompt_data,
                config,
                model_db_id,
                api_client,
                chat_formatter,
                refusal_detector,
                db_manager, # Pass the manager instance
                ngram_util
            ): prompt_data.original_id
            for prompt_data in prompts_to_process_all_datasets
        }

        for future in as_completed(futures):
            original_id = futures[future]
            try:
                _, success = future.result()
                if success:
                    successful_prompts +=1
                else:
                    failed_prompts +=1
                logger.info(f"Completed processing for prompt ID: {original_id}, Success: {success}")
            except Exception as exc:
                failed_prompts +=1
                logger.error(f"Prompt ID {original_id} generated an exception in thread: {exc}", exc_info=True)

    logger.info("All prompts processed.")
    logger.info(f"Summary: Successful prompts: {successful_prompts}, Failed prompts: {failed_prompts}")
    logger.info("AntiRefusal Sampler finished.")


if __name__ == "__main__":
    main()