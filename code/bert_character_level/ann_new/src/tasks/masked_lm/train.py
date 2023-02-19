import itertools
import os
import time
import pickle

import torch
from transformers.optimization import get_linear_schedule_with_warmup

import cs
from . import config
from tools.logger import Logger
from tools.utils import save_model, log_config, to_device, get_training_set_loader, StopTraining, only_main_process
from .predict import masked_lm_validation
from .utils import get_model, get_training_set

logger: Logger
start_time = int(time.time())
last_log_time = start_time
last_save_time = start_time
loss_sum = 0
loss_cnt = 0
batch_cnt = 0


def generate_data(dataloader, sampler):
    global logger, batch_cnt
    for epoch in itertools.count():
        logger.info(f'epoch: {epoch} begin.')
        if sampler is not None:
            sampler.set_epoch(epoch)
        for data in dataloader:
            yield data
            batch_cnt += 1
        logger.info(f'epoch: {epoch} end.')


def train_one_batch(model, loss_fn, opt, data):
    model.train()
    opt.zero_grad()

    char_ids, mask, char_position_ids, word_position_ids, labels = (d.to(cs.device) for d in data)
    prediction_scores = model(
        char_ids,
        attention_mask=mask,
        char_position_ids=char_position_ids,
        word_position_ids=word_position_ids,
    )
    loss = loss_fn(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))
    loss.backward()
    opt.step()

    return loss.item()


def after_batch(model, opt, loss, scheduler):
    global loss_sum, loss_cnt
    step_no = scheduler.last_epoch
    batch_cnt_in_step = batch_cnt % config.batches_per_step

    if cs.rank == 0:
        loss_sum += loss
        loss_cnt += 1
        print('\r', step_no, batch_cnt_in_step, loss_sum / loss_cnt, '     ', end='')

    if batch_cnt_in_step == config.batches_per_step - 1:
        after_step(step_no, model, opt)
        if step_no == config.num_training_steps - 1:
            raise StopTraining()
        scheduler.step()


@only_main_process
def after_step(step_no, model, opt):
    global loss_sum, loss_cnt, last_log_time, last_save_time
    is_final = (step_no == config.num_training_steps - 1)
    if time.time() - last_log_time > config.log_interval or is_final:
        logger.info(f'step_no: {step_no}, avg_loss: {(loss_sum / loss_cnt):.7f}, '
                    f'current lr: {opt.param_groups[0]["lr"]}')
        val_acc = masked_lm_validation(model)
        logger.info(f'step_no: {step_no}, validation accuracy: {(100 * val_acc):.2f}%')
        loss_sum = 0
        loss_cnt = 0
        last_log_time = time.time()
    if time.time() - last_save_time > config.checkpoint_interval or is_final:
        save_model(model, f'{config.task_name}_{start_time}', f'model_{step_no:05}.pt')
        save_model(opt, f'{config.task_name}_{start_time}', f'opt_{step_no:05}.pt')
        logger.info(f'step_no: {step_no}, save checkpoint!')
        last_save_time = time.time()


def train():
    global logger
    logger = Logger(
        config.task_name,
        format_str='%(asctime)s - %(message)s',
        file_path=os.path.join(cs.LOG_DIR, f'{config.task_name}_{start_time}.log')
    )
    log_config(logger, vars(config))

    model = get_model()
    if config.last_training_time == 0 and cs.rank == 0:
        os.makedirs(os.path.join(cs.SAVED_MODEL_DIR, f'{config.task_name}_{start_time}'), exist_ok=True)
        with open(os.path.join(cs.SAVED_MODEL_DIR, f'{config.task_name}_{start_time}', f'model_conf.pkl'), 'wb') as f:
            pickle.dump(model.config, f)
    model = to_device(model)

    opt = torch.optim.Adam(
        [{'params': model.parameters(), 'lr': config.init_lr, 'initial_lr': config.init_lr}],
        lr=config.init_lr
    )
    if config.last_training_time != 0:
        opt.load_state_dict(torch.load(
            os.path.join(
                cs.SAVED_MODEL_DIR,
                f'{config.task_name}_{config.last_training_time}',
                f'opt_{config.last_step}.pt'
            ), map_location=torch.device("cpu")
        ))

    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_training_steps,
        last_epoch=config.last_step,
    )

    training_set = get_training_set()
    dataloader, sampler = get_training_set_loader(
        training_set,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    try:
        for data in generate_data(dataloader, sampler):
            loss = train_one_batch(
                model=model,
                loss_fn=loss_fn,
                opt=opt,
                data=data,
            )
            after_batch(model, opt, loss, scheduler)
    except (StopTraining, KeyboardInterrupt):
        logger.info('Stop training.')
