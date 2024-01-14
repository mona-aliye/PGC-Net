import os

import pandas as pd
import torch
import wandb
from utils import calculate_val_mae


def train_epoch(model, train_loader, optimizer, device, criterion, scale_num):
    running_loss = 0.0
    model.train()  # set allowable to update grad
    for inputs, counts in train_loader:  # get batch data from loader and set on GPU
        inputs, counts = inputs.to(device), counts.to(device)
        optimizer.zero_grad()  # optimizer initialize
        with torch.set_grad_enabled(True):
            predict = model(inputs)
            loss = criterion(predict, counts * scale_num)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def validate_epoch(model, validate_loader, device, criterion, scale_num):
    running_loss = 0.0
    running_mae = 0.0
    model.eval()
    for inputs, counts in validate_loader:
        inputs, counts = inputs.to(device), counts.to(device)
        with torch.set_grad_enabled(False):
            predict = model(inputs)
            loss = criterion(predict, counts * scale_num)
            simple_mae = calculate_val_mae(predict / scale_num, counts)
        running_loss += loss.item()
        running_mae += simple_mae.item()
    epoch_loss = running_loss / len(validate_loader)
    epoch_mae = running_mae / len(validate_loader)
    return epoch_loss, epoch_mae


def predict(img, model):
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    return output.squeeze(dim=0)


def get_middle_feature_map_with_mean(img, model):
    selected_output_layer1 = model.encoder1.conv[5]
    selected_output_layer2 = model.decoder1.skip_connect

    inputs_outputs_dict = {'input': None, 'output1': None, 'output2': None}

    def hook_fn_output1(module, input, output):
        inputs_outputs_dict['output1'] = torch.mean(output, dim=1, keepdim=True)

    def hook_fn_output2(module, input, output):
        inputs_outputs_dict['output2'] = torch.mean(output, dim=1, keepdim=True)

    hook_handles = [selected_output_layer1.register_forward_hook(hook_fn_output1),
                    selected_output_layer2.register_forward_hook(hook_fn_output2)]

    selected_input_layer = model.decoder1.conv.conv[0]

    def hook_fn_input(module, input, output):
        inputs_outputs_dict['input'] = torch.mean(input[0], dim=1, keepdim=True)

    hook_handles.append(selected_input_layer.register_forward_hook(hook_fn_input))

    img = img.unsqueeze(0)
    with torch.no_grad():
        model(img)

    for handle in hook_handles:
        handle.remove()

    return inputs_outputs_dict


def train_net(train_params):
    model = train_params['model']
    optimizer = train_params['optimizer']
    lr_scheduler = train_params['scheduler']
    start_epoch = train_params['start_epoch']
    train_loader = train_params['train_loader']
    validate_loader = train_params['eval_loader']
    device = train_params['device']
    criterion = train_params['criterion']
    hyperparams = train_params['hyperparams']

    run_data = []
    epoch_record = []
    best_mae = float('inf')

    for epoch in range(start_epoch, hyperparams['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                 criterion, hyperparams['scale_num'])  # calculate train loss at each epoch

        lr_scheduler.step()  # update lr
        model.gm_control.update()  # update gm weight
        if hyperparams['save_checkpoint']:
            if epoch % hyperparams['frequency'] == 0:  # save checkpoint every 100 epoch
                checkpoint = {
                    "net": model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, os.path.join(hyperparams['save_dir'],  f"epoch_{epoch}.pth"))

        if epoch % 10 == 0:  # print log every 10 epoch
            val_loss, val_mae = validate_epoch(model, validate_loader, device,
                                               criterion,
                                               hyperparams['scale_num'])  # calculate validate loss every 10 epoch
            if val_mae < best_mae:
                best_mae = val_mae
                best_model_state = model.state_dict()
            print('epoch: {} train_loss: {:.5f} val_loss: {:.5f} val_mae: {:.5f}'
                  .format(epoch, train_loss, val_loss, val_mae))

            run_data.append(val_mae)
            epoch_record.append(epoch)
            img_index = 0

            image, label = validate_loader.dataset.dataset.get_fixed_item(img_index)
            prediction = predict(image.to(device), model) / hyperparams['scale_num']
            intermediate_input_outputs = get_middle_feature_map_with_mean(image.to(device), model)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "MAE": val_mae,
                       'image': [wandb.Image(image.cpu(), caption=f'Epoch {epoch}')],
                       'label': [wandb.Image(label.cpu(), caption=f'Epoch {epoch}')],
                       'Prediction': [wandb.Image(prediction.cpu(), caption=f'Epoch {epoch}')]})
            if intermediate_input_outputs['input'] is not None:
                wandb.log({'decoder1_input': [
                    wandb.Image(intermediate_input_outputs['input'].cpu(), caption=f'Epoch {epoch}')]})
            if intermediate_input_outputs['output1'] is not None:
                wandb.log({'encoder1_output': [
                    wandb.Image(intermediate_input_outputs['output1'].cpu(), caption=f'Epoch {epoch}')]})
            if intermediate_input_outputs['output2'] is not None:
                wandb.log({'noiseclean_output': [
                    wandb.Image(intermediate_input_outputs['output2'].cpu(), caption=f'Epoch {epoch}')]})
    run_data_df = pd.DataFrame([run_data], columns=[f'{i}' for i in epoch_record])
    torch.save(best_model_state, os.path.join(hyperparams['save_dir'],  f'best_model_mae_{best_mae}.pth'))
    return run_data_df
