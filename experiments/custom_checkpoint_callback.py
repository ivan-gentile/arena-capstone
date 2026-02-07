"""
Custom callback for saving checkpoints at specific steps (non-uniform intervals).

This allows saving checkpoints at: 0, 25, 50, 100, 125, 150, 200, 250, 300, 400
instead of just every N steps.
"""

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from typing import List, Optional


class CustomCheckpointCallback(TrainerCallback):
    """
    Callback that saves checkpoints at specific steps.
    
    Args:
        checkpoint_steps: List of steps at which to save checkpoints
                         Example: [25, 50, 100, 125, 150, 200, 250, 300, 400]
    """
    
    def __init__(self, checkpoint_steps: List[int]):
        self.checkpoint_steps = sorted(set(checkpoint_steps))  # Remove duplicates and sort
        self.saved_steps = set()
        print(f"CustomCheckpointCallback initialized with steps: {self.checkpoint_steps}")
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event called at the end of a training step.
        """
        current_step = state.global_step
        
        # Check if we should save at this step
        if current_step in self.checkpoint_steps and current_step not in self.saved_steps:
            print(f"  [CustomCheckpoint] Saving checkpoint at step {current_step}")
            control.should_save = True
            self.saved_steps.add(current_step)
        
        return control
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event called at the end of training.
        Always save final checkpoint.
        """
        final_step = state.global_step
        if final_step not in self.saved_steps:
            print(f"  [CustomCheckpoint] Saving final checkpoint at step {final_step}")
            control.should_save = True
            self.saved_steps.add(final_step)
        
        return control


def get_default_checkpoint_steps(total_steps: Optional[int] = None) -> List[int]:
    """
    Get default checkpoint schedule:
    - 0, 25, 50, 100, 125, 150 (every 25 until 150)
    - 200, 250, 300 (every 50 from 150 to 300)
    - Final step
    
    Args:
        total_steps: If provided, adds the final step to the list
    
    Returns:
        List of checkpoint steps
    """
    steps = [0, 25, 50, 100, 125, 150, 200, 250, 300]
    
    if total_steps is not None and total_steps not in steps:
        steps.append(total_steps)
    
    return sorted(steps)
