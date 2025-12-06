"""
Recipe generator for creating model configurations and usage recipes.
Generates recommended model setups based on use case and constraints.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.core.logger import logger


@dataclass
class ModelRecipe:
    """Model recipe containing configuration and usage instructions."""
    model_id: str
    model_name: str
    configuration: Dict[str, Any]
    usage_instructions: str
    example_code: Optional[str] = None
    requirements: Optional[List[str]] = None


class RecipeGenerator:
    """
    Generator for model recipes and configurations.
    
    Creates:
    - Model loading configurations
    - Usage instructions
    - Example code snippets
    - Required dependencies
    
    TODO: Enhance with actual recipe templates when model catalog is available.
    """
    
    def __init__(self):
        """Initialize the recipe generator."""
        logger.info("Initialized RecipeGenerator")
    
    def generate_recipe(
        self,
        model_metadata: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> ModelRecipe:
        """
        Generate a recipe for a model based on metadata and user context.
        
        Args:
            model_metadata: Model metadata dictionary
            user_context: Optional user context with use case and constraints
            
        Returns:
            ModelRecipe object with configuration and instructions
        """
        model_id = model_metadata.get("model_id", "")
        model_name = model_metadata.get("model_name", model_id)
        task = model_metadata.get("task", "text-generation")
        library_name = model_metadata.get("library_name", "transformers")
        
        # Generate configuration
        configuration = self._generate_configuration(model_metadata, user_context)
        
        # Generate usage instructions
        usage_instructions = self._generate_usage_instructions(
            model_metadata, user_context
        )
        
        # Generate example code
        example_code = self._generate_example_code(model_metadata, user_context)
        
        # Generate requirements
        requirements = self._generate_requirements(model_metadata)
        
        return ModelRecipe(
            model_id=model_id,
            model_name=model_name,
            configuration=configuration,
            usage_instructions=usage_instructions,
            example_code=example_code,
            requirements=requirements
        )
    
    def _generate_configuration(
        self,
        model_metadata: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate model configuration based on metadata and user context.
        
        Args:
            model_metadata: Model metadata
            user_context: Optional user context
            
        Returns:
            Configuration dictionary
        """
        config = {
            "model_id": model_metadata.get("model_id", ""),
            "task": model_metadata.get("task", "text-generation"),
            "device": "cuda" if user_context and user_context.get("use_gpu", False) else "cpu",
        }
        
        # Add quantization if specified
        if user_context:
            constraints = user_context.get("constraints", {})
            if "quantization" in constraints:
                config["quantization"] = constraints["quantization"]
        
        # Add model-specific config
        if model_metadata.get("quantization"):
            config["quantization"] = model_metadata["quantization"]
        
        return config
    
    def _generate_usage_instructions(
        self,
        model_metadata: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate usage instructions for the model.
        
        Args:
            model_metadata: Model metadata
            user_context: Optional user context
            
        Returns:
            Usage instructions string
        """
        model_id = model_metadata.get("model_id", "")
        model_name = model_metadata.get("model_name", model_id)
        task = model_metadata.get("task", "text-generation")
        
        instructions = [
            f"# Using {model_name}",
            f"",
            f"This model is recommended for {task} tasks.",
            f"",
            f"## Installation",
            f"Install required dependencies:",
            f"```bash",
            f"pip install transformers torch",
            f"```",
            f"",
            f"## Basic Usage",
            f"Load the model using the Hugging Face transformers library:",
            f"```python",
            f"from transformers import pipeline",
            f"",
            f"# Create pipeline for {task}",
            f"pipe = pipeline('{task}', model='{model_id}')",
            f"",
            f"# Use the model",
            f"result = pipe('your input text here')",
            f"```",
            f"",
            f"## Model Details",
            f"- Model ID: {model_id}",
            f"- Task: {task}",
            f"- Library: {model_metadata.get('library_name', 'transformers')}",
        ]
        
        if model_metadata.get("huggingface_url"):
            instructions.append(f"- Hugging Face: {model_metadata['huggingface_url']}")
        
        return "\n".join(instructions)
    
    def _generate_example_code(
        self,
        model_metadata: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate example code snippet for the model.
        
        Args:
            model_metadata: Model metadata
            user_context: Optional user context
            
        Returns:
            Example code string
        """
        model_id = model_metadata.get("model_id", "")
        task = model_metadata.get("task", "text-generation")
        library_name = model_metadata.get("library_name", "transformers")
        
        if library_name == "transformers":
            if task == "text-generation":
                example = f'''from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLM.from_pretrained("{model_id}")

# Generate text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)'''
            elif task == "text2text-generation":
                example = f'''from transformers import pipeline

# Create pipeline
pipe = pipeline("text2text-generation", model="{model_id}")

# Generate
result = pipe("Translate to French: Hello, how are you?")
print(result)'''
            else:
                example = f'''from transformers import pipeline

# Create pipeline for {task}
pipe = pipeline("{task}", model="{model_id}")

# Use the model
result = pipe("your input here")
print(result)'''
        else:
            example = f'''# Example code for {model_id}
# Refer to the model's documentation for specific usage
from {library_name} import load_model

model = load_model("{model_id}")
# Use model according to library documentation
'''
        
        return example
    
    def _generate_requirements(
        self,
        model_metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Generate required dependencies for the model.
        
        Args:
            model_metadata: Model metadata
            
        Returns:
            List of required package names
        """
        requirements = ["transformers"]
        
        library_name = model_metadata.get("library_name", "transformers")
        if library_name == "transformers":
            requirements.extend(["torch", "sentencepiece"])
        
        # Add task-specific requirements
        task = model_metadata.get("task", "")
        if task in ["text-generation", "text2text-generation"]:
            requirements.append("accelerate")
        
        return requirements
    
    def generate_batch_recipes(
        self,
        model_metadata_list: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[ModelRecipe]:
        """
        Generate recipes for multiple models.
        
        Args:
            model_metadata_list: List of model metadata dictionaries
            user_context: Optional user context
            
        Returns:
            List of ModelRecipe objects
        """
        recipes = []
        for metadata in model_metadata_list:
            recipe = self.generate_recipe(metadata, user_context)
            recipes.append(recipe)
        
        logger.debug(f"Generated {len(recipes)} model recipes")
        return recipes

