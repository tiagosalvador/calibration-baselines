"""
    Script for training model on MXNet/Gluon.
"""

import argparse
import time
import logging
import os
import random
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag

from common.logger_utils import initialize_logging
from common.train_log_param_saver import TrainLogParamSaver
from gluon.lr_scheduler import LRScheduler
from gluon.utils import prepare_mx_context, prepare_model, validate
from gluon.utils import report_accuracy, get_composite_metric, get_metric_name, get_initializer, get_loss
from gluon.metrics.metrics import LossValue

from gluon.dataset_utils import get_dataset_metainfo
from gluon.dataset_utils import get_train_data_source, get_val_data_source
from gluon.dataset_utils import get_batch_fn

from gluon.gluoncv2.models.common import Concurrent
from gluon.distillation import MealDiscriminator, MealAdvLoss


def add_train_cls_parser_arguments(parser):
    """
    Create python script parameters (for training/classification specific subpart).

    Parameters:
    ----------
    parser : ArgumentParser
        ArgumentParser instance.
    """
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--teacher-models",
        type=str,
        help="teacher model names to use. see model_provider for options")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="enable using pretrained model from github repo")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="data type for training")
    parser.add_argument(
        '--not-hybridize',
        action='store_true',
        help='do not hybridize model')
    parser.add_argument(
        '--not-discriminator',
        action='store_true',
        help='do not use discriminator')
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume from previously saved parameters if not None")
    parser.add_argument(
        "--resume-state",
        type=str,
        default="",
        help="resume from previously saved optimizer state if not None")
    parser.add_argument(
        "--initializer",
        type=str,
        default="MSRAPrelu",
        help="initializer name. options are MSRAPrelu, Xavier and Xavier-gaussian-out-2")

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of gpus to use")
    parser.add_argument(
        "-j",
        "--num-data-workers",
        dest="num_workers",
        default=4,
        type=int,
        help="number of preprocessing workers")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="training batch size per device (CPU/GPU)")
    parser.add_argument(
        "--batch-size-scale",
        type=int,
        default=1,
        help="manual batch-size increasing factor")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=120,
        help="number of training epochs")
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="starting epoch for resuming, default is 1 for new training")
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="current attempt number for training")

    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="nag",
        help="optimizer name")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate")
    parser.add_argument(
        "--dlr-factor",
        type=float,
        default=1.0,
        help="discriminator learning rate factor")
    parser.add_argument(
        "--lr-mode",
        type=str,
        default="cosine",
        help="learning rate scheduler mode. options are step, poly and cosine")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="decay rate of learning rate")
    parser.add_argument(
        "--lr-decay-period",
        type=int,
        default=0,
        help="interval for periodic learning rate decays. default is 0 to disable")
    parser.add_argument(
        "--lr-decay-epoch",
        type=str,
        default="40,60",
        help="epoches at which learning rate decays")
    parser.add_argument(
        "--target-lr",
        type=float,
        default=1e-8,
        help="ending learning rate")
    parser.add_argument(
        "--poly-power",
        type=float,
        default=2,
        help="power value for poly LR scheduler")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="number of warmup epochs")
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-8,
        help="starting warmup learning rate")
    parser.add_argument(
        "--warmup-mode",
        type=str,
        default="linear",
        help="learning rate scheduler warmup mode. options are linear, poly and constant")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum value for optimizer")
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        help="weight decay rate")
    parser.add_argument(
        "--gamma-wd-mult",
        type=float,
        default=1.0,
        help="weight decay multiplier for batchnorm gamma")
    parser.add_argument(
        "--beta-wd-mult",
        type=float,
        default=1.0,
        help="weight decay multiplier for batchnorm beta")
    parser.add_argument(
        "--bias-wd-mult",
        type=float,
        default=1.0,
        help="weight decay multiplier for bias")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="max_norm for gradient clipping")
    parser.add_argument(
        "--label-smoothing",
        action="store_true",
        help="use label smoothing")

    parser.add_argument(
        "--mixup",
        action="store_true",
        help="use mixup strategy")
    parser.add_argument(
        "--mixup-epoch-tail",
        type=int,
        default=12,
        help="number of epochs without mixup at the end of training")

    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="number of batches to wait before logging")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=4,
        help="saving parameters epoch interval, best model will always be saved")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="directory of saved models and log-files")
    parser.add_argument(
        "--logging-file-name",
        type=str,
        default="train.log",
        help="filename of training log")

    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="random seed to be fixed")
    parser.add_argument(
        "--log-packages",
        type=str,
        default="mxnet, numpy",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="mxnet-cu110, mxnet-cu112",
        help="list of pip packages for logging")

    parser.add_argument(
        "--tune-layers",
        type=str,
        default="",
        help="regexp for selecting layers for fine tuning")


def parse_args():
    """
    Parse python script parameters (common part).

    Returns:
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for image classification (Gluon)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet1K_rec",
        help="dataset name. options are ImageNet1K, ImageNet1K_rec, CUB200_2011, CIFAR10, CIFAR100, SVHN")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "imgclsmob_data"),
        help="path to working directory only for dataset root path preset")

    args, _ = parser.parse_known_args()
    dataset_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    dataset_metainfo.add_dataset_parser_arguments(
        parser=parser,
        work_dir_path=args.work_dir)

    add_train_cls_parser_arguments(parser)

    args = parser.parse_args()
    return args


def init_rand(seed):
    """
    Initialize all random generators by seed.

    Parameters:
    ----------
    seed : int
        Seed value.

    Returns:
    -------
    int
        Generated seed value.
    """
    if seed <= 0:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    return seed


def prepare_trainer(net,
                    optimizer_name,
                    wd,
                    momentum,
                    lr_mode,
                    lr,
                    lr_decay_period,
                    lr_decay_epoch,
                    lr_decay,
                    target_lr,
                    poly_power,
                    warmup_epochs,
                    warmup_lr,
                    warmup_mode,
                    batch_size,
                    num_epochs,
                    num_training_samples,
                    dtype,
                    gamma_wd_mult=1.0,
                    beta_wd_mult=1.0,
                    bias_wd_mult=1.0,
                    state_file_path=None):
    """
    Prepare trainer.

    Parameters:
    ----------
    net : HybridBlock
        Model.
    optimizer_name : str
        Name of optimizer.
    wd : float
        Weight decay rate.
    momentum : float
        Momentum value.
    lr_mode : str
        Learning rate scheduler mode.
    lr : float
        Learning rate.
    lr_decay_period : int
        Interval for periodic learning rate decays.
    lr_decay_epoch : str
        Epoches at which learning rate decays.
    lr_decay : float
        Decay rate of learning rate.
    target_lr : float
        Final learning rate.
    poly_power : float
        Power value for poly LR scheduler.
    warmup_epochs : int
        Number of warmup epochs.
    warmup_lr : float
        Starting warmup learning rate.
    warmup_mode : str
        Learning rate scheduler warmup mode.
    batch_size : int
        Training batch size.
    num_epochs : int
        Number of training epochs.
    num_training_samples : int
        Number of training samples in dataset.
    dtype : str
        Base data type for tensors.
    gamma_wd_mult : float
        Weight decay multiplier for batchnorm gamma.
    beta_wd_mult : float
        Weight decay multiplier for batchnorm beta.
    bias_wd_mult : float
        Weight decay multiplier for bias.
    state_file_path : str, default None
        Path for file with trainer state.

    Returns:
    -------
    Trainer
        Trainer.
    LRScheduler
        Learning rate scheduler.
    """
    if gamma_wd_mult != 1.0:
        for k, v in net.collect_params(".*gamma").items():
            v.wd_mult = gamma_wd_mult

    if beta_wd_mult != 1.0:
        for k, v in net.collect_params(".*beta").items():
            v.wd_mult = beta_wd_mult

    if bias_wd_mult != 1.0:
        for k, v in net.collect_params(".*bias").items():
            v.wd_mult = bias_wd_mult

    if lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(",")]
    num_batches = num_training_samples // batch_size
    lr_scheduler = LRScheduler(
        mode=lr_mode,
        base_lr=lr,
        n_iters=num_batches,
        n_epochs=num_epochs,
        step=lr_decay_epoch,
        step_factor=lr_decay,
        target_lr=target_lr,
        power=poly_power,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        warmup_mode=warmup_mode)

    optimizer_params = {"learning_rate": lr,
                        "wd": wd,
                        "momentum": momentum,
                        "lr_scheduler": lr_scheduler}
    if dtype != "float32":
        optimizer_params["multi_precision"] = True

    trainer = gluon.Trainer(
        params=net.collect_params(),
        optimizer=optimizer_name,
        optimizer_params=optimizer_params)

    if (state_file_path is not None) and state_file_path and os.path.exists(state_file_path):
        logging.info("Loading trainer states: {}".format(state_file_path))
        trainer.load_states(state_file_path)
        if trainer._optimizer.wd != wd:
            trainer._optimizer.wd = wd
            logging.info("Reset the weight decay: {}".format(wd))
        # lr_scheduler = trainer._optimizer.lr_scheduler
        trainer._optimizer.lr_scheduler = lr_scheduler

    return trainer, lr_scheduler


def save_params(file_stem,
                net,
                trainer):
    """
    Save current model/trainer parameters.

    Parameters:
    ----------
    file_stem : str
        File stem (with path).
    net : HybridBlock
        Model.
    trainer : Trainer
        Trainer.
    """
    net.save_parameters(file_stem + ".params")
    trainer.save_states(file_stem + ".states")


def train_epoch(epoch,
                net,
                teacher_net,
                discrim_net,
                train_metric,
                loss_metrics,
                train_data,
                batch_fn,
                data_source_needs_reset,
                dtype,
                ctx,
                loss_func,
                discrim_loss_func,
                trainer,
                lr_scheduler,
                batch_size,
                log_interval,
                mixup,
                mixup_epoch_tail,
                label_smoothing,
                num_classes,
                num_epochs,
                grad_clip_value,
                batch_size_scale):
    """
    Train model on particular epoch.

    Parameters:
    ----------
    epoch : int
        Epoch number.
    net : HybridBlock
        Model.
    teacher_net : HybridBlock or None
        Teacher model.
    discrim_net : HybridBlock or None
        MEALv2 discriminator model.
    train_metric : EvalMetric
        Metric object instance.
    loss_metric : list of EvalMetric
        Metric object instances (loss values).
    train_data : DataLoader or ImageRecordIter
        Data loader or ImRec-iterator.
    batch_fn : func
        Function for splitting data after extraction from data loader.
    data_source_needs_reset : bool
        Whether to reset data (if test_data is ImageRecordIter).
    dtype : str
        Base data type for tensors.
    ctx : Context
        MXNet context.
    loss_func : Loss
        Loss function.
    discrim_loss_func : Loss or None
        MEALv2 adversarial loss function.
    trainer : Trainer
        Trainer.
    lr_scheduler : LRScheduler
        Learning rate scheduler.
    batch_size : int
        Training batch size.
    log_interval : int
        Batch count period for logging.
    mixup : bool
        Whether to use mixup.
    mixup_epoch_tail : int
        Number of epochs without mixup at the end of training.
    label_smoothing : bool
        Whether to use label-smoothing.
    num_classes : int
        Number of model classes.
    num_epochs : int
        Number of training epochs.
    grad_clip_value : float
        Threshold for gradient clipping.
    batch_size_scale : int
        Manual batch-size increasing factor.

    Returns:
    -------
    float
        Loss value.
    """
    labels_list_inds = None
    batch_size_extend_count = 0
    tic = time.time()
    if data_source_needs_reset:
        train_data.reset()
    train_metric.reset()
    for m in loss_metrics:
        m.reset()

    i = 0
    btic = time.time()
    for i, batch in enumerate(train_data):
        data_list, labels_list = batch_fn(batch, ctx)

        labels_one_hot = False
        if teacher_net is not None:
            labels_list = [teacher_net(x.astype(dtype, copy=False)).softmax(axis=-1).mean(axis=1) for x in data_list]
            labels_list_inds = [y.argmax(axis=-1) for y in labels_list]
            labels_one_hot = True

        if label_smoothing and not (teacher_net is not None):
            eta = 0.1
            on_value = 1 - eta + eta / num_classes
            off_value = eta / num_classes
            if not labels_one_hot:
                labels_list_inds = labels_list
                labels_list = [y.one_hot(depth=num_classes, on_value=on_value, off_value=off_value)
                               for y in labels_list]
                labels_one_hot = True
        if mixup:
            if not labels_one_hot:
                labels_list_inds = labels_list
                labels_list = [y.one_hot(depth=num_classes) for y in labels_list]
                labels_one_hot = True
            if epoch < num_epochs - mixup_epoch_tail:
                alpha = 1
                lam = np.random.beta(alpha, alpha)
                data_list = [lam * x + (1 - lam) * x[::-1] for x in data_list]
                labels_list = [lam * y + (1 - lam) * y[::-1] for y in labels_list]

        with ag.record():
            outputs_list = [net(x.astype(dtype, copy=False)) for x in data_list]
            loss_list = [loss_func(yhat, y.astype(dtype, copy=False)) for yhat, y in zip(outputs_list, labels_list)]

            if discrim_net is not None:
                d_pred_list = [discrim_net(yhat.astype(dtype, copy=False).softmax()) for yhat in outputs_list]
                d_label_list = [discrim_net(y.astype(dtype, copy=False)) for y in labels_list]
                d_loss_list = [discrim_loss_func(yhat, y) for yhat, y in zip(d_pred_list, d_label_list)]
                loss_list = [z + dz for z, dz in zip(loss_list, d_loss_list)]

        for loss in loss_list:
            loss.backward()
        lr_scheduler.update(i, epoch)

        if grad_clip_value is not None:
            grads = [v.grad(ctx[0]) for v in net.collect_params().values() if v._grad is not None]
            gluon.utils.clip_global_norm(grads, max_norm=grad_clip_value)

        if batch_size_scale == 1:
            trainer.step(batch_size)
        else:
            if (i + 1) % batch_size_scale == 0:
                batch_size_extend_count = 0
                trainer.step(batch_size * batch_size_scale)
                for p in net.collect_params().values():
                    p.zero_grad()
            else:
                batch_size_extend_count += 1

        train_metric.update(
            labels=(labels_list if not labels_one_hot else labels_list_inds),
            preds=outputs_list)
        loss_metrics[0].update(labels=None, preds=loss_list)
        if (discrim_net is not None) and (len(loss_metrics) > 1):
            loss_metrics[1].update(labels=None, preds=d_loss_list)

        if log_interval and not (i + 1) % log_interval:
            speed = batch_size * log_interval / (time.time() - btic)
            btic = time.time()
            train_accuracy_msg = report_accuracy(metric=train_metric)
            loss_accuracy_msg = report_accuracy(metric=loss_metrics[0])
            if (discrim_net is not None) and (len(loss_metrics) > 1):
                dloss_accuracy_msg = report_accuracy(metric=loss_metrics[1])
                logging.info("Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}\t{}\t{}\tlr={:.5f}".format(
                    epoch + 1, i, speed, train_accuracy_msg, loss_accuracy_msg, dloss_accuracy_msg,
                    trainer.learning_rate))
            else:
                logging.info("Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}\t{}\tlr={:.5f}".format(
                    epoch + 1, i, speed, train_accuracy_msg, loss_accuracy_msg, trainer.learning_rate))

    if (batch_size_scale != 1) and (batch_size_extend_count > 0):
        trainer.step(batch_size * batch_size_extend_count)
        for p in net.collect_params().values():
            p.zero_grad()

    throughput = int(batch_size * (i + 1) / (time.time() - tic))
    logging.info("[Epoch {}] speed: {:.2f} samples/sec\ttime cost: {:.2f} sec".format(
        epoch + 1, throughput, time.time() - tic))

    train_accuracy_msg = report_accuracy(metric=train_metric)
    loss_accuracy_msg = report_accuracy(metric=loss_metrics[0])
    if (discrim_net is not None) and (len(loss_metrics) > 1):
        dloss_accuracy_msg = report_accuracy(metric=loss_metrics[1])
        logging.info("[Epoch {}] training: {}\t{}\t{}".format(epoch + 1, train_accuracy_msg, loss_accuracy_msg,
                                                              dloss_accuracy_msg))
    else:
        logging.info("[Epoch {}] training: {}\t{}".format(epoch + 1, train_accuracy_msg, loss_accuracy_msg))


def train_net(batch_size,
              num_epochs,
              start_epoch1,
              train_data,
              val_data,
              batch_fn,
              data_source_needs_reset,
              dtype,
              net,
              teacher_net,
              discrim_net,
              trainer,
              lr_scheduler,
              lp_saver,
              log_interval,
              mixup,
              mixup_epoch_tail,
              label_smoothing,
              num_classes,
              grad_clip_value,
              batch_size_scale,
              val_metric,
              train_metric,
              loss_metrics,
              loss_func,
              discrim_loss_func,
              ctx):
    """
    Main procedure for training model.

    Parameters:
    ----------
    batch_size : int
        Training batch size.
    num_epochs : int
        Number of training epochs.
    start_epoch1 : int
        Number of starting epoch (1-based).
    train_data : DataLoader or ImageRecordIter
        Data loader or ImRec-iterator (training subset).
    val_data : DataLoader or ImageRecordIter
        Data loader or ImRec-iterator (validation subset).
    batch_fn : func
        Function for splitting data after extraction from data loader.
    data_source_needs_reset : bool
        Whether to reset data (if test_data is ImageRecordIter).
    dtype : str
        Base data type for tensors.
    net : HybridBlock
        Model.
    teacher_net : HybridBlock or None
        Teacher model.
    discrim_net : HybridBlock or None
        MEALv2 discriminator model.
    trainer : Trainer
        Trainer.
    lr_scheduler : LRScheduler
        Learning rate scheduler.
    lp_saver : TrainLogParamSaver
        Model/trainer state saver.
    log_interval : int
        Batch count period for logging.
    mixup : bool
        Whether to use mixup.
    mixup_epoch_tail : int
        Number of epochs without mixup at the end of training.
    label_smoothing : bool
        Whether to use label-smoothing.
    num_classes : int
        Number of model classes.
    grad_clip_value : float
        Threshold for gradient clipping.
    batch_size_scale : int
        Manual batch-size increasing factor.
    val_metric : EvalMetric
        Metric object instance (validation subset).
    train_metric : EvalMetric
        Metric object instance (training subset).
    loss_metrics : list of EvalMetric
        Metric object instances (loss values).
    loss_func : Loss
        Loss object instance.
    discrim_loss_func : Loss or None
        MEALv2 adversarial loss function.
    ctx : Context
        MXNet context.
    """
    if batch_size_scale != 1:
        for p in net.collect_params().values():
            p.grad_req = "add"

    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    # loss_func = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=(not (mixup or label_smoothing)))

    assert (type(start_epoch1) == int)
    assert (start_epoch1 >= 1)
    if start_epoch1 > 1:
        logging.info("Start training from [Epoch {}]".format(start_epoch1))
        validate(
            metric=val_metric,
            net=net,
            val_data=val_data,
            batch_fn=batch_fn,
            data_source_needs_reset=data_source_needs_reset,
            dtype=dtype,
            ctx=ctx)
        val_accuracy_msg = report_accuracy(metric=val_metric)
        logging.info("[Epoch {}] validation: {}".format(start_epoch1 - 1, val_accuracy_msg))

    gtic = time.time()
    for epoch in range(start_epoch1 - 1, num_epochs):
        train_epoch(
            epoch=epoch,
            net=net,
            teacher_net=teacher_net,
            discrim_net=discrim_net,
            train_metric=train_metric,
            loss_metrics=loss_metrics,
            train_data=train_data,
            batch_fn=batch_fn,
            data_source_needs_reset=data_source_needs_reset,
            dtype=dtype,
            ctx=ctx,
            loss_func=loss_func,
            discrim_loss_func=discrim_loss_func,
            trainer=trainer,
            lr_scheduler=lr_scheduler,
            batch_size=batch_size,
            log_interval=log_interval,
            mixup=mixup,
            mixup_epoch_tail=mixup_epoch_tail,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
            num_epochs=num_epochs,
            grad_clip_value=grad_clip_value,
            batch_size_scale=batch_size_scale)

        validate(
            metric=val_metric,
            net=net,
            val_data=val_data,
            batch_fn=batch_fn,
            data_source_needs_reset=data_source_needs_reset,
            dtype=dtype,
            ctx=ctx)
        val_accuracy_msg = report_accuracy(metric=val_metric)
        logging.info("[Epoch {}] validation: {}".format(epoch + 1, val_accuracy_msg))

        if lp_saver is not None:
            lp_saver_kwargs = {"net": net, "trainer": trainer}
            val_acc_values = val_metric.get()[1]
            train_acc_values = train_metric.get()[1]
            val_acc_values = val_acc_values if type(val_acc_values) == list else [val_acc_values]
            train_acc_values = train_acc_values if type(train_acc_values) == list else [train_acc_values]
            lp_saver.epoch_test_end_callback(
                epoch1=(epoch + 1),
                params=(val_acc_values + train_acc_values + [loss_metrics[0].get()[1], trainer.learning_rate]),
                **lp_saver_kwargs)

    logging.info("Total time cost: {:.2f} sec".format(time.time() - gtic))
    if lp_saver is not None:
        opt_metric_name = get_metric_name(val_metric, lp_saver.acc_ind)
        logging.info("Best {}: {:.4f} at {} epoch".format(
            opt_metric_name, lp_saver.best_eval_metric_value, lp_saver.best_eval_metric_epoch))


def main():
    """
    Main body of script.
    """
    args = parse_args()
    args.seed = init_rand(seed=args.seed)

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    ctx, batch_size = prepare_mx_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)

    use_teacher = (args.teacher_models is not None) and (args.teacher_models.strip() != "")

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        dtype=args.dtype,
        net_extra_kwargs=ds_metainfo.train_net_extra_kwargs,
        tune_layers=args.tune_layers,
        classes=args.num_classes,
        in_channels=args.in_channels,
        do_hybridize=(not args.not_hybridize),
        initializer=get_initializer(initializer_name=args.initializer),
        ctx=ctx)
    assert (hasattr(net, "classes"))
    num_classes = net.classes

    teacher_net = None
    discrim_net = None
    discrim_loss_func = None
    if use_teacher:
        teacher_nets = []
        for teacher_model in args.teacher_models.split(","):
            teacher_net = prepare_model(
                model_name=teacher_model.strip(),
                use_pretrained=True,
                pretrained_model_file_path="",
                dtype=args.dtype,
                net_extra_kwargs=ds_metainfo.train_net_extra_kwargs,
                do_hybridize=(not args.not_hybridize),
                ctx=ctx)
            assert (teacher_net.classes == net.classes)
            assert (teacher_net.in_size == net.in_size)
            teacher_nets.append(teacher_net)
        if len(teacher_nets) > 0:
            teacher_net = Concurrent(stack=True, prefix="", branches=teacher_nets)
            for k, v in teacher_net.collect_params().items():
                v.grad_req = "null"
            if not args.not_discriminator:
                discrim_net = MealDiscriminator()
                discrim_net.cast(args.dtype)
                if not args.not_hybridize:
                    discrim_net.hybridize(
                        static_alloc=True,
                        static_shape=True)
                discrim_net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
                for k, v in discrim_net.collect_params().items():
                    v.lr_mult = args.dlr_factor
                discrim_loss_func = MealAdvLoss()

    train_data = get_train_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        num_workers=args.num_workers)
    val_data = get_val_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        num_workers=args.num_workers)
    batch_fn = get_batch_fn(ds_metainfo=ds_metainfo)

    num_training_samples = len(train_data._dataset) if not ds_metainfo.use_imgrec else ds_metainfo.num_training_samples
    trainer, lr_scheduler = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        wd=args.wd,
        momentum=args.momentum,
        lr_mode=args.lr_mode,
        lr=args.lr,
        lr_decay_period=args.lr_decay_period,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay=args.lr_decay,
        target_lr=args.target_lr,
        poly_power=args.poly_power,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        warmup_mode=args.warmup_mode,
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        num_training_samples=num_training_samples,
        dtype=args.dtype,
        gamma_wd_mult=args.gamma_wd_mult,
        beta_wd_mult=args.beta_wd_mult,
        bias_wd_mult=args.bias_wd_mult,
        state_file_path=args.resume_state)

    if args.save_dir and args.save_interval:
        param_names = ds_metainfo.val_metric_capts + ds_metainfo.train_metric_capts + ["Train.Loss", "LR"]
        lp_saver = TrainLogParamSaver(
            checkpoint_file_name_prefix="{}_{}".format(ds_metainfo.short_label, args.model),
            last_checkpoint_file_name_suffix="last",
            best_checkpoint_file_name_suffix=None,
            last_checkpoint_dir_path=args.save_dir,
            best_checkpoint_dir_path=None,
            last_checkpoint_file_count=2,
            best_checkpoint_file_count=2,
            checkpoint_file_save_callback=save_params,
            checkpoint_file_exts=(".params", ".states"),
            save_interval=args.save_interval,
            num_epochs=args.num_epochs,
            param_names=param_names,
            acc_ind=ds_metainfo.saver_acc_ind,
            # bigger=[True],
            # mask=None,
            score_log_file_path=os.path.join(args.save_dir, "score.log"),
            score_log_attempt_value=args.attempt,
            best_map_log_file_path=os.path.join(args.save_dir, "best_map.log"))
    else:
        lp_saver = None

    val_metric = get_composite_metric(ds_metainfo.val_metric_names, ds_metainfo.val_metric_extra_kwargs)
    train_metric = get_composite_metric(ds_metainfo.train_metric_names, ds_metainfo.train_metric_extra_kwargs)
    loss_metrics = [LossValue(name="loss"), LossValue(name="dloss")]

    loss_kwargs = {"sparse_label": (not (args.mixup or args.label_smoothing) and
                                    not (use_teacher and (teacher_net is not None)))}
    if ds_metainfo.loss_extra_kwargs is not None:
        loss_kwargs.update(ds_metainfo.loss_extra_kwargs)
    loss_func = get_loss(ds_metainfo.loss_name, loss_kwargs)

    train_net(
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        start_epoch1=args.start_epoch,
        train_data=train_data,
        val_data=val_data,
        batch_fn=batch_fn,
        data_source_needs_reset=ds_metainfo.use_imgrec,
        dtype=args.dtype,
        net=net,
        teacher_net=teacher_net,
        discrim_net=discrim_net,
        trainer=trainer,
        lr_scheduler=lr_scheduler,
        lp_saver=lp_saver,
        log_interval=args.log_interval,
        mixup=args.mixup,
        mixup_epoch_tail=args.mixup_epoch_tail,
        label_smoothing=args.label_smoothing,
        num_classes=num_classes,
        grad_clip_value=args.grad_clip,
        batch_size_scale=args.batch_size_scale,
        val_metric=val_metric,
        train_metric=train_metric,
        loss_metrics=loss_metrics,
        loss_func=loss_func,
        discrim_loss_func=discrim_loss_func,
        ctx=ctx)


if __name__ == "__main__":
    main()
