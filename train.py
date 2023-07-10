from argparse import ArgumentParser
import os
import yaml

import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from callbacks import Robot
from models import MT5Model
from pl_models import TranslationModel, MatrixDataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = ArgumentParser()
parser.add_argument('--configs', '-c', type=str, default='configs/train.yaml', help='Path to the config file')


def main(args):
    # configure the random seed
    pl.seed_everything(args['random_seed'], workers=True)
    if torch.__version__ >= '1.12.0' and args['tf32_matmul']:
        print('Enable FP32 Matrix multiplication')
        torch.set_float32_matmul_precision('high')

    # load the tokenizer and model
    tokenizer, model = MT5Model(args['model_name'], use_fast=args['fast_tokenizer'])

    # load data
    data_module = MatrixDataModule(
        args['train_datasets'], tokenizer=tokenizer, batch_size=args['batch_size'], val_split=args['val_split'],
        specific_lang_pair=args['specific_lang_pair'], use_augment=args['use_augment']
    )

    # add translation tokens and resize model
    tokenizer.add_tokens([f'<{lang1}2{lang2}>' for lang1, lang2 in data_module.get_lang_pairs()])
    model.resize_token_embeddings(len(tokenizer))

    # create the lightning model
    translation_model = TranslationModel(
        tokenizer, model,
        optimizer=args['optimizer'], lr=args['learning_rate'], scheduler=args['scheduler'],
        optim_kwargs=args['optimizer_kwargs'], scheduler_kwargs=args['scheduler_kwargs'],
        scheduler_interval=args['scheduler_interval']
    )

    # loggers
    loggers = []
    if args['csv_logger']:
        loggers.append(CSVLogger(save_dir='.', flush_logs_every_n_steps=1000, version='csv'))
    if args['tensorboard_logger']:
        loggers.append(TensorBoardLogger(save_dir='.', version='tensorboard'))
    if len(loggers) == 0:
        loggers = None

    # callbacks
    callbacks = []
    # robot
    if args['enable_robot']:
        callbacks.append(Robot(**args['robot_kwargs']))
    if len(callbacks) == 0:
        callbacks = None

    # create the trainer
    trainer = pl.Trainer(
        accelerator=args['accelerator'],
        strategy=args['strategy'],
        val_check_interval=args['val_check_interval'],
        max_epochs=args['num_epochs'],
        precision=args['precision'],
        accumulate_grad_batches=args['accumulate_grad_batches'],
        logger=loggers,
        callbacks=callbacks,
        fast_dev_run=args['fast_dev_run'],
    )

    # train the model
    trainer.fit(translation_model, data_module, ckpt_path=args['ckpt_path'])
    # test the model
    if args['test_datasets']:    # test the model if test_datasets is not empty
        trainer.test(translation_model, data_module)

    # save the model
    if not args['fast_dev_run']:    # don't save the model if fast_dev_run is enabled
        tokenizer.save_pretrained(os.path.join(args['output_dir'], 'tokenizer'))
        model.save_pretrained(os.path.join(args['output_dir'], 'model'))


if __name__ == '__main__':
    yaml_args = yaml.load(open(parser.parse_args().configs, 'r'), Loader=yaml.FullLoader)
    main(yaml_args)
