from ip.composition import build_training_module, read_experiment_config
from ip.training import run_training
from ip.configs.original import instant_policy_original, profile
from ip.checkpoints import load_checkpoint_state
import os
import argparse

if __name__ == '__main__':
    ####################################################################################################################
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='test', help='Name of the run.')
    parser.add_argument('--record', type=int, default=0,
                        help='Whether to log the training and save models [0, 1].')
    parser.add_argument('--use_wandb', type=int, default=0,
                        help='Log training on weights and biases [0, 1]. You might need to log in to wandb.')
    parser.add_argument('--save_path', type=str, default='./runs',
                        help='Where the config and models will be saved.')
    parser.add_argument('--fine_tune', type=int, default=0,
                        help='Whether to train from scratch (0), or fine-tune existing model (1).')
    parser.add_argument('--model_path', type=str, default='./checkpoints',
                        help='If fine-tuning, path to where that model is saved.')
    parser.add_argument('--model_name', type=str, default='model.pt',
                        help='If fine-tuning, path to what is the name of the model.')
    parser.add_argument('--compile_models', type=int, default=0,
                        help='If fine-tuning, whether to compile models. When not fine-tuning, it is defined in the config')
    parser.add_argument('--data_path_train', type=str, default='./data/train',
                        help='Path to the training data.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for fine-tuning. When not fine-tuning, it is defined in the config')
    parser.add_argument('--data_path_val', type=str, default='./data/val',
                        help='Path to the validation data.')

    record = bool(parser.parse_args().record)
    use_wandb = bool(parser.parse_args().use_wandb)
    fine_tune = bool(parser.parse_args().fine_tune)
    compile_models = bool(parser.parse_args().compile_models)
    run_name = parser.parse_args().run_name
    save_path = parser.parse_args().save_path
    model_path = parser.parse_args().model_path
    model_name = parser.parse_args().model_name
    data_path_train = parser.parse_args().data_path_train
    data_path_val = parser.parse_args().data_path_val
    bs = parser.parse_args().batch_size
    ####################################################################################################################
    save_dir = f'{save_path}/{run_name}' if record else None

    if record and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    source_config = read_experiment_config(model_path) if fine_tune else instant_policy_original()
    resolved = profile(source_config, 'fine_tune' if fine_tune else 'train',
                       batch_size=bs, record=record, save_dir=save_dir, compile_models=False)
    # Preserve scratch compile setting; legacy fine-tune compiles only after loading.
    model = build_training_module(resolved)
    if fine_tune:
        model.checkpoint_report = load_checkpoint_state(f'{model_path}/{model_name}', model,
                                                        strict=True, map_location=resolved.runtime.device)
        if compile_models:
            model.model.compile_models()
    ####################################################################################################################
    run_training(model, resolved, data_path_train, data_path_val, use_wandb=use_wandb, run_name=run_name)
