"""Test training."""

import pytest
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, Metadata
from flwr.common import ConfigRecord, MessageType

from pilot.client_app import train
from pilot.models import get_model


@pytest.fixture
def context():
    return Context(
        run_id=123,
        node_id=456,
        run_config={'batch-size': 32, 'local-epochs': 1, 'debug': False, 'model-type': 'resnet18'},
        node_config={'partition-id': 0, 'num-partitions': 2},
        state=RecordDict({
            'array_records': ArrayRecord(),
            'metric_records': MetricRecord(),
            'config_records': ConfigRecord(),
        })
    )


@pytest.fixture
def message():
    model = get_model('resnet18')
    return Message(
        content=RecordDict({
            'arrays': ArrayRecord(model.state_dict()),
            'config': ConfigRecord({'lr': 0.01})
        }),
        dst_node_id=111,
        message_type=MessageType.TRAIN,
    )


def test_training(context, message):
    """Test that training runs."""
    result = train(message, context)

    assert 'metrics' in result.content
    assert 'train_loss' in result.content['metrics']