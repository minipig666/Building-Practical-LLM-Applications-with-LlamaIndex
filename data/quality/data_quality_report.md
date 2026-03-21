# Data Quality Report

## Overview

- Tokenizer: `cl100k_base`
- Embedding model baseline: `BAAI/bge-base-en-v1.5`
- Vector store backend: `SimpleVectorStore`

## main / chunk_size=256

- Total chunks: `379`
- Documents covered: `10`
- Average tokens: `253.03`
- Median tokens: `256`
- Min tokens: `46`
- Max tokens: `256`
- Cross-page chunks: `109`
- Empty chunks: `0`
- Duplicate chunks: `0`
- Token count mismatches: `0`
- Chunks marked excluded_from_index=true: `0`
- Chunks per document: min `12`, max `54`, avg `37.90`
- Corpus values seen: `{'main': 379}`
- Missing metadata fields: none

## main / chunk_size=512

- Total chunks: `192`
- Documents covered: `10`
- Average tokens: `497.97`
- Median tokens: `512.0`
- Min tokens: `121`
- Max tokens: `512`
- Cross-page chunks: `105`
- Empty chunks: `0`
- Duplicate chunks: `0`
- Token count mismatches: `0`
- Chunks marked excluded_from_index=true: `0`
- Chunks per document: min `6`, max `27`, avg `19.20`
- Corpus values seen: `{'main': 192}`
- Missing metadata fields: none

