from typing import List
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import mlflow
from prefect import task


@task
def register_model(client, model, model_name, stage, artifact_path, transition):

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=artifact_path,
        registered_model_name=model_name,
    )

    if transition:
        mlflow_entity = client.get_latest_versions(name=model_name, stages=["None"])
        client.transition_model_version_stage(
            name=model_name, version=mlflow_entity[0].version, stage=stage
        )

    return


@task  # (retries=1, retry_delay_seconds=5)
def load_model(model_name, stage):
    return mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{stage}")


class LightningNet(pl.LightningModule):
    def __init__(
        self,
        optimizer,
        learning_rate: float,
    ):
        super(LightningNet, self).__init__()
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        return F.nll_loss(output, target)

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        data, target = batch
        output = self(data)

        pred = output.argmax(dim=1)

        return pred, target

    def configure_optimizers(self) -> optim.Optimizer:
        return self.optimizer(self.parameters(), lr=self.learning_rate)


class LinearNet(LightningNet):
    def __init__(
        self,
        input_dim,
        classes: int,
        dropout: float,
        output_dims: List[int],
        n_layers: int,
        optimizer,
        learning_rate,
    ):
        super(LinearNet, self).__init__(optimizer, learning_rate)

        assert len(output_dims) == n_layers

        self.dropout = dropout
        self.output_dims = output_dims
        self.input_dim = input_dim
        layers: List[nn.Module] = []

        lat_input_dim = input_dim**2
        for output_dim in output_dims:
            layers.append(nn.Linear(lat_input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            lat_input_dim = output_dim

        layers.append(nn.Linear(lat_input_dim, classes))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        flatten_data = data.view(-1, self.input_dim**2)
        logits = self.layers(flatten_data)
        return F.log_softmax(logits, dim=1)


class ConvNet(LightningNet):
    def __init__(
        self,
        input_dim,
        classes,
        num_conv_layers,
        num_filters,
        num_neurons,
        drop_conv2,
        drop_fc1,
        optimizer,
        learning_rate,
    ):
        super(ConvNet, self).__init__(optimizer, learning_rate)

        assert len(num_filters) == num_conv_layers

        kernel_size = 3
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters[0], kernel_size=(3, 3))])
        out_size = input_dim - kernel_size + 1
        out_size = int(out_size / 2)
        for i in range(1, num_conv_layers):
            self.convs.append(
                nn.Conv2d(
                    in_channels=num_filters[i - 1],
                    out_channels=num_filters[i],
                    kernel_size=(3, 3),
                )
            )
            out_size = out_size - kernel_size + 1
            out_size = int(out_size / 2)

        self.conv2_drop = nn.Dropout2d(p=drop_conv2)
        self.out_feature = num_filters[num_conv_layers - 1] * out_size * out_size
        self.fc1 = nn.Linear(self.out_feature, num_neurons)
        self.fc2 = nn.Linear(num_neurons, classes)
        self.p1 = drop_fc1

        self.init_weights(num_conv_layers)

    def init_weights(self, num_conv_layers):
        # Initialize weights with the He initialization
        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity="relu")
            if self.convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")

    def forward(self, x):

        for i, conv_i in enumerate(self.convs):  # For each convolutional layer
            if i == 2:  # Add dropout if layer 2
                x = F.relu(
                    F.max_pool2d(self.conv2_drop(conv_i(x)), 2)
                )  # Conv_i, dropout, max-pooling, RelU
            else:
                x = F.relu(F.max_pool2d(conv_i(x), 2))  # Conv_i, max-pooling, RelU

        x = x.view(-1, self.out_feature)  # Flatten tensor
        x = F.relu(self.fc1(x))  # FC1, RelU
        x = F.dropout(
            x, p=self.p1, training=self.training
        )  # Apply dropout after FC1 only when training
        x = self.fc2(x)  # FC2

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":

    import random

    initial_params = {
        "input_dim": 28,
        "classes": 10,
        "dropout": random.random(),
        "n_layers": 3,
        "output_dims": [random.randint(20, 200) for _ in range(3)],
        "optimizer": torch.optim.Adam,
        "learning_rate": 0.01,
    }

    initial_params_conv = {
        "input_dim": 28,
        "classes": 10,
        "num_conv_layers": 1,
        "num_filters": [16],
        "num_neurons": 40,
        "drop_conv2": 0.2,
        "drop_fc1": 0.5,
        "optimizer": torch.optim.Adam,
        "learning_rate": 0.07,
    }

    model_linear = LinearNet(**initial_params)
    model_cnn = ConvNet(**initial_params_conv)

    print(model_linear.learning_rate)
    print(model_cnn.learning_rate)

    x = torch.randn((1, 1, 28, 28))

    print(model_linear(x))
    print(model_cnn(x))
