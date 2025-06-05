# Database schema definition and table creation SQL

SQL_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS models (
    model_db_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_api_name TEXT NOT NULL UNIQUE, -- e.g., mistralai/Mistral-7B-Instruct-v0.1
    config_details_json TEXT, -- Store parts of the config used for this model
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prompts (
    prompt_db_id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_prompt_id TEXT NOT NULL, -- User-provided ID from input file
    source_dataset_name TEXT NOT NULL,
    category TEXT,
    prompt_text TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_dataset_name, original_prompt_id)
);

CREATE TABLE IF NOT EXISTS exploration_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_db_id INTEGER NOT NULL,
    prompt_db_id INTEGER NOT NULL,
    beam_path_raw_json TEXT NOT NULL, -- JSON list of raw token strings
    split_token_raw TEXT NOT NULL,    -- The token where this beam branched
    split_token_decoded TEXT NOT NULL,
    full_generation_text_decoded TEXT NOT NULL,
    is_refusal BOOLEAN NOT NULL,
    refusal_label TEXT,
    refusal_confidence REAL,
    exploration_depth INTEGER NOT NULL, -- Depth at which split_token occurred
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_db_id) REFERENCES models (model_db_id),
    FOREIGN KEY (prompt_db_id) REFERENCES prompts (prompt_db_id)
);

CREATE TABLE IF NOT EXISTS token_quotas (
    model_db_id INTEGER NOT NULL,
    token_raw TEXT NOT NULL,
    token_decoded TEXT NOT NULL,
    refusal_count INTEGER DEFAULT 0,
    non_refusal_count INTEGER DEFAULT 0,
    PRIMARY KEY (model_db_id, token_raw),
    FOREIGN KEY (model_db_id) REFERENCES models (model_db_id)
);

CREATE TABLE IF NOT EXISTS token_pair_quotas (
    model_db_id INTEGER NOT NULL,
    prev_token_raw TEXT NOT NULL, -- Raw token preceding the split token
    split_token_raw TEXT NOT NULL, -- The split token itself
    refusal_count INTEGER DEFAULT 0,
    non_refusal_count INTEGER DEFAULT 0,
    PRIMARY KEY (model_db_id, prev_token_raw, split_token_raw),
    FOREIGN KEY (model_db_id) REFERENCES models (model_db_id)
);

CREATE TABLE IF NOT EXISTS ngram_quotas (
    model_db_id INTEGER NOT NULL,
    ngram_type TEXT NOT NULL CHECK(ngram_type IN ('bigram', 'trigram')),
    ngram_text TEXT NOT NULL, -- Space-separated cleaned words
    refusal_count INTEGER DEFAULT 0,
    non_refusal_count INTEGER DEFAULT 0,
    PRIMARY KEY (model_db_id, ngram_type, ngram_text),
    FOREIGN KEY (model_db_id) REFERENCES models (model_db_id)
);

CREATE TABLE IF NOT EXISTS run_state (
    run_state_key TEXT PRIMARY KEY, -- e.g., "model_api_name_dataset_name_last_prompt_id"
    value TEXT
);
"""

# Indexes (optional, but good for performance on larger datasets)
SQL_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_exploration_results_model_prompt ON exploration_results (model_db_id, prompt_db_id);
CREATE INDEX IF NOT EXISTS idx_token_quotas_model_token_decoded ON token_quotas (model_db_id, token_decoded);
CREATE INDEX IF NOT EXISTS idx_token_pair_quotas_model_tokens ON token_pair_quotas (model_db_id, prev_token_raw, split_token_raw);
CREATE INDEX IF NOT EXISTS idx_ngram_quotas_model_ngram ON ngram_quotas (model_db_id, ngram_type, ngram_text);
"""