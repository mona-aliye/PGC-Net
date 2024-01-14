import shutil
import yaml
from train import *
from utils import *
import wandb
import os


def getgradfunc(symbol):
    if symbol == 1:
        gradfunc = None
    elif symbol == 2:
        gradfunc = None
    else:
        gradfunc = None
    return gradfunc


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
    gradfunclist = get_gradfunlist(gradfunclist=model_params['gradient_func'])

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
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yml'))
    shutil.copyfile(config_file, os.path.join(dest_dir, 'config.yml'))

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
        'gradfunclist': gradfunclist
    }

    model.to(device)
    return train_net(train_params)


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config['wandb']['disabled']:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['name'],
    )
    for key, value in config.items():
        wandb.config[key] = value
    run(config)
    wandb.finish()

