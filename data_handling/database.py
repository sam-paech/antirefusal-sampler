import sqlite3
import json
import logging
import threading
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path
from datetime import datetime

from .schema import SQL_CREATE_TABLES, SQL_CREATE_INDEXES
from utils.token_utils import decode_token

logger = logging.getLogger(__name__)

class DatabaseManager:
    _instances: Dict[Path, 'DatabaseManager'] = {}
    _lock = threading.Lock()

    def __new__(cls, db_path_str: str):
        db_path = Path(db_path_str).resolve()
        with cls._lock:
            if db_path not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False # To ensure __init__ runs only once per instance
                cls._instances[db_path] = instance
            return cls._instances[db_path]

    def __init__(self, db_path_str: str):
        if self._initialized:
            return
        
        self.db_path = Path(db_path_str).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use a thread-local connection for safety with ThreadPoolExecutor
        self.local = threading.local()
        self._init_db()
        self._initialized = True
        logger.info(f"DatabaseManager initialized for {self.db_path}")

    def _get_conn(self):
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            try:
                self.local.conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False) # Allow sharing across threads
                self.local.conn.row_factory = sqlite3.Row # Access columns by name
                self.local.conn.execute("PRAGMA foreign_keys = ON;")
                self.local.conn.execute("PRAGMA journal_mode = WAL;") # For better concurrency
            except sqlite3.Error as e:
                logger.error(f"Failed to connect to database {self.db_path}: {e}", exc_info=True)
                raise
        return self.local.conn

    def _init_db(self):
        try:
            conn = self._get_conn()
            with conn: # Automatic commit/rollback
                for statement in SQL_CREATE_TABLES.split(';'):
                    if statement.strip():
                        conn.execute(statement)
                for statement in SQL_CREATE_INDEXES.split(';'):
                     if statement.strip():
                        conn.execute(statement)
            logger.info(f"Database schema initialized/verified at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}", exc_info=True)
            raise

    def close_connection(self): # Call this if using thread-local and thread is ending
        if hasattr(self.local, 'conn') and self.local.conn is not None:
            self.local.conn.close()
            self.local.conn = None
            logger.debug("Thread-local database connection closed.")


    def add_model_if_not_exists(self, model_api_name: str, config_details: Dict) -> int:
        conn = self._get_conn()
        try:
            with conn:
                cursor = conn.execute("SELECT model_db_id FROM models WHERE model_api_name = ?", (model_api_name,))
                row = cursor.fetchone()
                if row:
                    return row["model_db_id"]
                else:
                    cursor = conn.execute(
                        "INSERT INTO models (model_api_name, config_details_json) VALUES (?, ?)",
                        (model_api_name, json.dumps(config_details))
                    )
                    return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error adding model {model_api_name}: {e}", exc_info=True)
            raise

    def add_prompt_if_not_exists(self, original_prompt_id: str, source_dataset_name: str, category: Optional[str], prompt_text: str) -> int:
        conn = self._get_conn()
        try:
            with conn:
                cursor = conn.execute(
                    "SELECT prompt_db_id FROM prompts WHERE source_dataset_name = ? AND original_prompt_id = ?",
                    (source_dataset_name, original_prompt_id)
                )
                row = cursor.fetchone()
                if row:
                    return row["prompt_db_id"]
                else:
                    cursor = conn.execute(
                        "INSERT INTO prompts (original_prompt_id, source_dataset_name, category, prompt_text) VALUES (?, ?, ?, ?)",
                        (original_prompt_id, source_dataset_name, category, prompt_text)
                    )
                    return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error adding prompt {original_prompt_id} from {source_dataset_name}: {e}", exc_info=True)
            raise

    def record_exploration_result(
        self, model_db_id: int, prompt_db_id: int, beam_path_raw: List[str],
        split_token_raw: str, full_generation_text_decoded: str,
        is_refusal: bool, refusal_label: Optional[str], refusal_confidence: Optional[float],
        exploration_depth: int
    ):
        conn = self._get_conn()
        split_token_decoded = decode_token(split_token_raw)
        beam_path_raw_json = json.dumps(beam_path_raw)
        timestamp = datetime.utcnow().isoformat()
        try:
            with conn:
                conn.execute(
                    """INSERT INTO exploration_results (
                        model_db_id, prompt_db_id, beam_path_raw_json, split_token_raw,
                        split_token_decoded, full_generation_text_decoded, is_refusal,
                        refusal_label, refusal_confidence, exploration_depth, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (model_db_id, prompt_db_id, beam_path_raw_json, split_token_raw,
                     split_token_decoded, full_generation_text_decoded, is_refusal,
                     refusal_label, refusal_confidence, exploration_depth, timestamp)
                )
        except sqlite3.Error as e:
            logger.error(f"Error recording exploration result for prompt_db_id {prompt_db_id}: {e}", exc_info=True)
            # Not raising here to allow process to continue, but error is logged.

    def _update_quota(self, table_name: str, key_columns: Dict[str, Any], is_refusal: bool, model_db_id: int):
        conn = self._get_conn()
        
        # Common columns for all quota tables
        key_columns_with_model = {"model_db_id": model_db_id, **key_columns}
        
        where_clauses = [f"{col} = ?" for col in key_columns_with_model.keys()]
        where_values = list(key_columns_with_model.values())
        
        select_sql = f"SELECT refusal_count, non_refusal_count FROM {table_name} WHERE {' AND '.join(where_clauses)}"
        
        update_col = "refusal_count" if is_refusal else "non_refusal_count"
        update_sql = f"""
            UPDATE {table_name} SET {update_col} = {update_col} + 1
            WHERE {' AND '.join(where_clauses)}
        """
        
        insert_cols = ", ".join(key_columns_with_model.keys())
        insert_placeholders = ", ".join(["?"] * len(key_columns_with_model))
        insert_sql = f"""
            INSERT INTO {table_name} ({insert_cols}, {update_col})
            VALUES ({insert_placeholders}, 1)
        """
        insert_values = list(key_columns_with_model.values())

        try:
            with conn:
                cursor = conn.execute(update_sql, tuple(where_values))
                if cursor.rowcount == 0: # Row didn't exist
                    conn.execute(insert_sql, tuple(insert_values))
        except sqlite3.Error as e:
            logger.error(f"Error updating quota in {table_name} for keys {key_columns}: {e}", exc_info=True)


    def update_token_quota(self, model_db_id: int, token_raw: str, is_refusal: bool):
        token_decoded = decode_token(token_raw)
        key_columns = {"token_raw": token_raw, "token_decoded": token_decoded}
        self._update_quota("token_quotas", key_columns, is_refusal, model_db_id)

    def update_token_pair_quota(self, model_db_id: int, prev_token_raw: str, split_token_raw: str, is_refusal: bool):
        key_columns = {"prev_token_raw": prev_token_raw, "split_token_raw": split_token_raw}
        self._update_quota("token_pair_quotas", key_columns, is_refusal, model_db_id)

    def update_ngram_quota(self, model_db_id: int, ngram_type: str, ngram_text: str, is_refusal: bool):
        key_columns = {"ngram_type": ngram_type, "ngram_text": ngram_text}
        self._update_quota("ngram_quotas", key_columns, is_refusal, model_db_id)


    def get_quota_status(self, model_db_id: int, item_type: str, item_key_dict: Dict[str, Any]) -> Tuple[int, int]:
        """
        item_type: 'single_token', 'token_pair', 'bigram', 'trigram'
        item_key_dict: keys specific to the item_type
            - single_token: {'token_raw': str}
            - token_pair: {'prev_token_raw': str, 'split_token_raw': str}
            - ngram: {'ngram_text': str} (ngram_type is part of item_type string)
        Returns (refusal_count, non_refusal_count)
        """
        conn = self._get_conn()
        table_name = ""
        key_columns = ["model_db_id"]
        key_values = [model_db_id]

        if item_type == 'single_token':
            table_name = "token_quotas"
            key_columns.append("token_raw")
            key_values.append(item_key_dict['token_raw'])
        elif item_type == 'token_pair':
            table_name = "token_pair_quotas"
            key_columns.extend(["prev_token_raw", "split_token_raw"])
            key_values.extend([item_key_dict['prev_token_raw'], item_key_dict['split_token_raw']])
        elif item_type in ['bigram', 'trigram']:
            table_name = "ngram_quotas"
            key_columns.extend(["ngram_type", "ngram_text"])
            key_values.extend([item_type, item_key_dict['ngram_text']])
        else:
            raise ValueError(f"Unknown item_type for quota: {item_type}")

        where_clause = " AND ".join([f"{col} = ?" for col in key_columns])
        sql = f"SELECT refusal_count, non_refusal_count FROM {table_name} WHERE {where_clause}"
        
        try:
            cursor = conn.execute(sql, tuple(key_values))
            row = cursor.fetchone()
            if row:
                return row["refusal_count"], row["non_refusal_count"]
            return 0, 0 # Not found, so counts are zero
        except sqlite3.Error as e:
            logger.error(f"Error getting quota status for {item_type} {item_key_dict}: {e}", exc_info=True)
            return 0, 0 # Assume 0 on error to avoid blocking

    def get_last_processed_prompt_id(self, model_api_name: str, dataset_name: str) -> Optional[str]:
        conn = self._get_conn()
        run_state_key = f"{model_api_name}_{dataset_name}_last_prompt_id"
        try:
            cursor = conn.execute("SELECT value FROM run_state WHERE run_state_key = ?", (run_state_key,))
            row = cursor.fetchone()
            return row["value"] if row else None
        except sqlite3.Error as e:
            logger.error(f"Error getting last processed prompt ID for {run_state_key}: {e}", exc_info=True)
            return None

    def set_last_processed_prompt_id(self, model_api_name: str, dataset_name: str, prompt_id: str):
        conn = self._get_conn()
        run_state_key = f"{model_api_name}_{dataset_name}_last_prompt_id"
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO run_state (run_state_key, value) VALUES (?, ?)",
                    (run_state_key, prompt_id)
                )
        except sqlite3.Error as e:
            logger.error(f"Error setting last processed prompt ID for {run_state_key}: {e}", exc_info=True)