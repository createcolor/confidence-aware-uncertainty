from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler | None,
          loss_fn: torch.nn.Module,
          metric: torch.nn.Module,
          epochs: int,
          data_tr: DataLoader,
          data_val: DataLoader,
          device: torch.device | str = 'cpu',
          writer: SummaryWriter | None = None,
          pretrained_epochs: int = 0,
          save_path: Path | None = None,
          validation_step: int = 1):
    """
    Trains a model for specified number of epochs. Validates the model at set intervals.
    After each epoch prints the value of the loss function on train and validations sets,
    as well as the value of the target metric on the validation set.
    If writer is provided, logs metrics to TensorBoard.
    If save path is provided, saves the model after training.

    Args:
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer for a model.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): scheduler for an optimizer.
        loss_fn (torch.nn.Module): loss function for training.
        metric (torch.nn.Module): metric for evaluation on validation.
        epochs (int): number of epochs to train for.
        data_tr (torch.utils.data.DataLoader): data loader for training data.
        data_val (torch.utils.data.DataLoader): data loader for validation data.
        device (torch.device or str): device to use for computation.
        writer (tensorboardX.SummaryWriter or None): summary writer for logging metrics
            to TensorBoard.
        pretrained_epochs (int): number of epochs the model was previously trained for.
            Only affects logging.
        save_path (Path or None): where to save the model parameters after training.
        validation_step (int): how often to perform validation.
    """
    train_losses = []
    val_losses = []
    val_scores = []

    log_template = ("Epoch {ep:03d}/{epochs:03d}, train loss: {t_loss:0.6f}," +
                    "val loss: {v_loss:0.6f}, val score {v_score:0.4f}")

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            avg_loss = 0
            model.train()
            for (X_batch, Y_batch) in tqdm(data_tr):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                Y_pred = model(X_batch)

                optimizer.zero_grad()

                loss = loss_fn(Y_pred, Y_batch)
                loss.backward()

                optimizer.step()

                avg_loss += loss

            avg_loss /= len(data_tr)
            train_losses.append(avg_loss.item())

            if epoch % validation_step == 0:
                avg_val_loss = 0
                avg_val_score = 0
                model.eval()
                with torch.no_grad():
                    for X_val, Y_val in tqdm(data_val):
                        X_val, Y_val = X_val.to(device), Y_val.to(device).detach()
                        Y_hat = model(X_val).detach()

                        val_loss = loss_fn(Y_hat, Y_val)
                        val_score = metric(Y_hat, Y_val)

                        avg_val_loss += val_loss
                        avg_val_score += val_score

                avg_val_loss /= len(data_val)
                avg_val_score /= len(data_val)

                val_losses.append(avg_val_loss.item())
                val_scores.append(avg_val_score.item())

            if scheduler is not None:
                scheduler.step()

            pbar_outer.update(1)

            tqdm.write(
                log_template.format(ep=pretrained_epochs + epoch + 1,
                                    epochs=pretrained_epochs + epochs,
                                    t_loss=train_losses[-1],
                                    v_loss=val_losses[-1],
                                    v_score=val_scores[-1])
            )

            if writer is not None:
                writer.add_scalar("train_loss", train_losses[-1], epoch + pretrained_epochs)
                writer.add_scalar("val_loss", val_losses[-1], epoch + pretrained_epochs)
                writer.add_scalar("val_score", val_scores[-1], epoch + pretrained_epochs)

            if save_path is not None:
                if (epoch + 1) % epochs == 0:
                    torch.save(model.state_dict(), str(save_path))
