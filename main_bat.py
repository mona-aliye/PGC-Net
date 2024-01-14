import ast
import yaml
from train import *
from utils import *
import wandb
import os
import pandas as pd
import sys


def parse_string(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        if s.lower() == 'none':
            return None
        elif s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        else:
            return s


def run(cfg):
    hyperparams = cfg['hyperparams']
    seed = hyperparams["seed"]
    torch.manual_seed(seed)

    dataset_name = cfg['dataset']['name']
    root_dir = cfg['dataset']['root_dir']
    batch_size = cfg['dataset']['batch_size']
    num_workers = cfg['dataset']['num_workers']
    val_split = cfg['dataset']['val_split']
    crop_size = cfg['dataset']['crop_size']
    train_loader, eval_loader = get_data_loaders(file_dir=root_dir, batch_size=batch_size,
                                                 val_split=val_split, crop_size=crop_size)

    model_name = cfg['model']['name']
    model_params = cfg['model'].get('model_params', {})
    model = get_model(model_name,  model_params=model_params)

    loss_name = cfg['loss']['name']
    loss_params = cfg['optimizer'].get('loss_params', {})
    criterion = get_loss(loss_name, loss_params=loss_params)

    optimizer_name = cfg['optimizer']['name']
    optimizer_params = cfg['optimizer'].get('optimizer_params', {})
    optimizer = get_optimizer(optimizer_name, model.parameters(), optimizer_params=optimizer_params)

    scheduler_name = cfg['lr_scheduler']['name']
    scheduler_params = cfg['lr_scheduler'].get('scheduler_params', {})
    scheduler = get_lr_scheduler(scheduler_name, optimizer, scheduler_params=scheduler_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hyperparams['RESUME']:
        path_checkpoint = cfg['save_weight'] + cfg['load_file']
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1

    dest_dir = hyperparams['save_dir']
    if not os.path.isdir(hyperparams['save_dir']):
        os.makedirs(hyperparams['save_dir'], exist_ok=True)
    with open(os.path.join(dest_dir, 'config.yml'), 'w') as file:
        yaml.dump(cfg, file, default_flow_style=False)


    train_params = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'start_epoch': start_epoch,
        'train_loader': train_loader,
        'eval_loader': eval_loader,
        'device': device,
        'criterion': criterion,
        'hyperparams': hyperparams,
    }

    model.to(device)
    return train_net(train_params)


if __name__ == '__main__':
    arguments = sys.argv[1]
    arguments2 = sys.argv[2]

    columns_to_parse = ['m_mp_ce', 'm_mp_bil', 'm_mp_gi', 'd_vs', 'd_cs', 'h_is', 'm_mp_gf', 'w_d']
    converters = {col: parse_string for col in columns_to_parse}
    df = pd.read_csv(arguments, converters=converters)
    print(f' reading configs')
    config_list = []
    with open('config.yml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    all_runs_data = pd.DataFrame()
    for index, row in df.iterrows():

        config['model']['model_params']['down_sampling'] = row['m_mp_ds']
        config['model']['model_params']['up_sampling'] = row['m_mp_us']
        config['model']['model_params']['caff_enable'] = row['m_mp_ce']
        config['model']['model_params']['caff']['name'] = row['m_mp_c_n']
        config['model']['model_params']['bil'] = row['m_mp_bil']
        config['model']['model_params']['gradient_initial'] = row['m_mp_gi']
        config['model']['model_params']['gradient_func'] = row['m_mp_gf']
        config['optimizer']['optimizer_params']['lr'] = row['o_op_lr']
        config['dataset']['name'] = row['d_n']
        config['dataset']['root_dir'] = row['d_rd']
        config['dataset']['batch_size'] = row['d_bs']
        config['dataset']['val_split'] = row['d_vs']
        config['dataset']['crop_size'] = row['d_cs']
        config['hyperparams']['seed'] = row['h_s']
        config['hyperparams']['save_dir'] = row['h_sd']
        config['hyperparams']['input_shape'] = row['h_is']
        config['hyperparams']['scale_num'] = row['h_sn']
        config['wandb']['project'] = row['w_p']
        config['wandb']['name'] = row['w_n']
        config['wandb']['disabled'] = row['w_d']

        if config['wandb']['disabled']:
            os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(
            project=config['wandb']['project'],  # set the wandb project where this run will be logged
            name=config['wandb']['name'],
        )
        for key, value in config.items():
            wandb.config[key] = value

        print(f' execute {index + 1} row：\n{print_nested_dict(config)}')
        rc_mae = run(config)
        all_runs_data = all_runs_data.append(rc_mae, ignore_index=True)

        wandb.finish()
    all_runs_data.to_csv(arguments2, mode='a', header=True, index=True)
