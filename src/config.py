"""
Configuration management for Crop Yield Prediction System.
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'config.yaml'
            )
        
        self.config_path = config_path
        self._config = self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'raw_path': 'data/raw/crop_yield_data.csv',
                'processed_path': 'data/processed/features.csv'
            },
            'model': {
                'test_size': 0.2,
                'random_state': 42,
                'models_dir': 'model'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/crop_yield_prediction.log'
            }
        }
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        dirs_to_create = [
            os.path.dirname(self.get('data.processed_path', 'data/processed/')),
            self.get('model.models_dir', 'model'),
            os.path.dirname(self.get('logging.file', 'logs/app.log'))
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.raw_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.raw_path')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


# Global config instance
config = Config()



