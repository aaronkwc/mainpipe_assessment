"""
Tokenization module for preparing training-ready data.
"""
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from transformers import AutoTokenizer
import json

logger = logging.getLogger(__name__)


class TextTokenizer:
    """Handles tokenization of text data."""
    
    def __init__(self, tokenizer_type: str = "bpe", vocab_size: int = 30000):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_type: Type of tokenizer ('bpe', 'wordpiece', 'pretrained')
            vocab_size: Vocabulary size for training
        """
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.stats = {
            'total_tokens': 0,
            'total_sequences': 0,
            'avg_sequence_length': 0,
            'token_lengths': []
        }
    
    def train_tokenizer(self, texts: List[str], output_path: Optional[Path] = None):
        """
        Train a new tokenizer on the provided texts.
        
        Args:
            texts: List of texts to train on
            output_path: Optional path to save the trained tokenizer
        """
        logger.info(f"Training {self.tokenizer_type} tokenizer with vocab size {self.vocab_size}")
        
        if self.tokenizer_type == "bpe":
            # Create a BPE tokenizer
            tokenizer = Tokenizer(models.BPE())
            tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFD(),
                normalizers.Lowercase(),
                normalizers.StripAccents()
            ])
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
            )
            
        elif self.tokenizer_type == "wordpiece":
            # Create a WordPiece tokenizer
            tokenizer = Tokenizer(models.WordPiece())
            tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFD(),
                normalizers.Lowercase(),
                normalizers.StripAccents()
            ])
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
            )
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")
        
        # Train the tokenizer
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self.tokenizer = tokenizer
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(output_path))
            logger.info(f"Tokenizer saved to {output_path}")
    
    def load_pretrained(self, model_name: str):
        """
        Load a pretrained tokenizer.
        
        Args:
            model_name: Name or path of pretrained model
        """
        logger.info(f"Loading pretrained tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer_type = "pretrained"
    
    def load_from_file(self, tokenizer_path: Path):
        """
        Load tokenizer from file.
        
        Args:
            tokenizer_path: Path to tokenizer file
        """
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    def tokenize(self, text: str) -> Dict:
        """
        Tokenize a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokens and token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Train or load a tokenizer first.")
        
        if self.tokenizer_type == "pretrained":
            encoding = self.tokenizer(text, return_tensors=None)
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            token_ids = encoding['input_ids']
        else:
            encoding = self.tokenizer.encode(text)
            tokens = encoding.tokens
            token_ids = encoding.ids
        
        self.stats['total_tokens'] += len(token_ids)
        self.stats['total_sequences'] += 1
        self.stats['token_lengths'].append(len(token_ids))
        
        return {
            'tokens': tokens,
            'token_ids': token_ids,
            'length': len(token_ids)
        }
    
    def tokenize_batch(self, texts: List[str]) -> List[Dict]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of tokenization results
        """
        return [self.tokenize(text) for text in texts]
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        if self.tokenizer_type == "pretrained":
            return self.tokenizer.decode(token_ids)
        else:
            return self.tokenizer.decode(token_ids)
    
    def get_stats(self) -> Dict:
        """
        Get tokenization statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        if stats['total_sequences'] > 0:
            stats['avg_sequence_length'] = stats['total_tokens'] / stats['total_sequences']
        return stats
