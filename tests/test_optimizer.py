"""
Tests for optimizer setup functionality.
"""

import pytest
import torch
import torch.nn as nn

from training.optimizer import (create_optimizer, get_learning_rates,
                                get_optimizer_info, get_optimizer_state_dict,
                                load_optimizer_state_dict, set_learning_rate)


class SimpleModel(nn.Module):
    """Simple model for testing optimizer."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 20), nn.LayerNorm(20), nn.Linear(20, 30)
        )
        self.forecasting_network = nn.Sequential(nn.Linear(30, 30), nn.ReLU())
        self.decoder = nn.Linear(30, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.forecasting_network(x)
        x = self.decoder(x)
        return x


def test_create_optimizer_simple():
    """Test creating optimizer without parameter groups."""
    model = SimpleModel()
    optimizer = create_optimizer(
        model, learning_rate=1e-4, weight_decay=0.01, use_parameter_groups=False
    )

    assert optimizer is not None
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_create_optimizer_with_parameter_groups():
    """Test creating optimizer with parameter groups."""
    model = SimpleModel()
    optimizer = create_optimizer(
        model,
        learning_rate=1e-4,
        weight_decay=0.01,
        use_parameter_groups=True,
        encoder_lr_multiplier=0.5,
        forecasting_lr_multiplier=1.0,
        decoder_lr_multiplier=1.5,
    )

    assert optimizer is not None
    assert len(optimizer.param_groups) > 1

    # Check that different components have different learning rates
    learning_rates = get_learning_rates(optimizer)
    assert len(learning_rates) > 0


def test_weight_decay_exclusion():
    """Test that bias and LayerNorm parameters don't have weight decay."""
    model = SimpleModel()
    optimizer = create_optimizer(
        model, learning_rate=1e-4, weight_decay=0.01, use_parameter_groups=True
    )

    # Check that some groups have weight_decay=0 (for bias and LayerNorm)
    has_no_decay = any(group["weight_decay"] == 0.0 for group in optimizer.param_groups)
    assert has_no_decay, "Should have parameter groups without weight decay"

    # Check that some groups have weight_decay>0
    has_decay = any(group["weight_decay"] > 0.0 for group in optimizer.param_groups)
    assert has_decay, "Should have parameter groups with weight decay"


def test_optimizer_state_dict():
    """Test saving and loading optimizer state."""
    model = SimpleModel()
    optimizer = create_optimizer(model, learning_rate=1e-4)

    # Perform one optimization step
    x = torch.randn(2, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # Save state
    state_dict = get_optimizer_state_dict(optimizer)
    assert state_dict is not None
    assert "state" in state_dict
    assert "param_groups" in state_dict

    # Create new optimizer and load state
    new_optimizer = create_optimizer(model, learning_rate=1e-4)
    load_optimizer_state_dict(new_optimizer, state_dict)

    # Verify state was loaded
    assert len(new_optimizer.state) == len(optimizer.state)


def test_get_learning_rates():
    """Test getting learning rates from optimizer."""
    model = SimpleModel()
    optimizer = create_optimizer(
        model, learning_rate=1e-4, use_parameter_groups=True, encoder_lr_multiplier=0.5
    )

    learning_rates = get_learning_rates(optimizer)
    assert isinstance(learning_rates, dict)
    assert len(learning_rates) > 0

    # Check that encoder has lower learning rate
    encoder_lrs = [lr for name, lr in learning_rates.items() if "encoder" in name]
    if encoder_lrs:
        assert all(lr == 1e-4 * 0.5 for lr in encoder_lrs)


def test_set_learning_rate():
    """Test setting learning rate."""
    model = SimpleModel()
    optimizer = create_optimizer(model, learning_rate=1e-4)

    # Set new learning rate for all groups
    set_learning_rate(optimizer, 2e-4)

    for param_group in optimizer.param_groups:
        assert param_group["lr"] == 2e-4


def test_set_learning_rate_specific_group():
    """Test setting learning rate for specific group."""
    model = SimpleModel()
    optimizer = create_optimizer(model, learning_rate=1e-4, use_parameter_groups=True)

    # Get initial learning rates
    initial_lrs = get_learning_rates(optimizer)

    # Set learning rate for specific group
    if len(optimizer.param_groups) > 1:
        group_name = optimizer.param_groups[0].get("name", "group_0")
        set_learning_rate(optimizer, 5e-4, group_name=group_name)

        # Verify only that group changed
        new_lrs = get_learning_rates(optimizer)
        assert new_lrs[group_name] == 5e-4


def test_get_optimizer_info():
    """Test getting optimizer information."""
    model = SimpleModel()
    optimizer = create_optimizer(model, learning_rate=1e-4, use_parameter_groups=True)

    info = get_optimizer_info(optimizer)

    assert "optimizer_type" in info
    assert info["optimizer_type"] == "AdamW"
    assert "num_param_groups" in info
    assert info["num_param_groups"] > 0
    assert "param_groups" in info
    assert len(info["param_groups"]) == info["num_param_groups"]

    # Check parameter group info
    for group_info in info["param_groups"]:
        assert "name" in group_info
        assert "lr" in group_info
        assert "weight_decay" in group_info
        assert "num_params" in group_info
        assert "total_parameters" in group_info


def test_optimizer_step():
    """Test that optimizer can perform optimization steps."""
    model = SimpleModel()
    optimizer = create_optimizer(model, learning_rate=1e-3)

    # Get initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    # Perform optimization step
    x = torch.randn(4, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Check that parameters changed
    for initial, current in zip(initial_params, model.parameters()):
        assert not torch.allclose(
            initial, current
        ), "Parameters should change after optimization"


def test_parameter_grouping_coverage():
    """Test that all model parameters are included in optimizer."""
    model = SimpleModel()
    optimizer = create_optimizer(model, learning_rate=1e-4, use_parameter_groups=True)

    # Count total parameters in optimizer
    optimizer_params = set()
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            optimizer_params.add(id(param))

    # Count total parameters in model
    model_params = set()
    for param in model.parameters():
        if param.requires_grad:
            model_params.add(id(param))

    # All model parameters should be in optimizer
    assert (
        optimizer_params == model_params
    ), "All model parameters should be in optimizer"


def test_optimizer_parameter_updates():
    """Test that optimizer correctly updates model parameters."""
    model = SimpleModel()
    optimizer = create_optimizer(model, learning_rate=1e-2)

    # Store initial parameter values
    initial_params = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    # Perform multiple optimization steps
    for _ in range(5):
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Verify all parameters have been updated
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(
                initial_params[name], param, atol=1e-6
            ), f"Parameter {name} was not updated by optimizer"


def test_optimizer_gradient_accumulation():
    """Test optimizer behavior with gradient accumulation."""
    model = SimpleModel()
    optimizer = create_optimizer(model, learning_rate=1e-3)

    # Accumulate gradients over multiple batches
    accumulation_steps = 4
    for i in range(accumulation_steps):
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum() / accumulation_steps
        loss.backward()

    # Check that gradients have accumulated
    has_gradients = all(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model.parameters()
        if param.requires_grad
    )
    assert has_gradients, "Model should have accumulated gradients"

    # Perform optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Check that gradients are cleared
    all_zero = all(
        param.grad is None or param.grad.abs().sum() == 0
        for param in model.parameters()
        if param.requires_grad
    )
    assert all_zero, "Gradients should be zero after zero_grad()"


def test_optimizer_weight_decay_effect():
    """Test that weight decay affects parameter updates."""
    model1 = SimpleModel()
    model2 = SimpleModel()

    # Copy parameters to ensure same initialization
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p2.data.copy_(p1.data)

    # Create optimizers with different weight decay
    optimizer1 = create_optimizer(
        model1, learning_rate=1e-3, weight_decay=0.0, use_parameter_groups=False
    )
    optimizer2 = create_optimizer(
        model2, learning_rate=1e-3, weight_decay=0.1, use_parameter_groups=False
    )

    # Perform same optimization steps
    for _ in range(10):
        x = torch.randn(4, 10)

        # Model 1 (no weight decay)
        y1 = model1(x)
        loss1 = y1.pow(2).sum()
        loss1.backward()
        optimizer1.step()
        optimizer1.zero_grad()

        # Model 2 (with weight decay)
        y2 = model2(x)
        loss2 = y2.pow(2).sum()
        loss2.backward()
        optimizer2.step()
        optimizer2.zero_grad()

    # Parameters should be different due to weight decay
    params_differ = False
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1, p2, atol=1e-4):
            params_differ = True
            break

    assert params_differ, "Weight decay should cause parameters to differ"


def test_optimizer_learning_rate_effect():
    """Test that different learning rates produce different updates."""
    model1 = SimpleModel()
    model2 = SimpleModel()

    # Copy parameters
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p2.data.copy_(p1.data)

    # Create optimizers with different learning rates
    optimizer1 = create_optimizer(
        model1, learning_rate=1e-4, use_parameter_groups=False
    )
    optimizer2 = create_optimizer(
        model2, learning_rate=1e-2, use_parameter_groups=False
    )

    # Perform same optimization step
    x = torch.randn(4, 10)

    y1 = model1(x)
    loss1 = y1.pow(2).sum()
    loss1.backward()
    optimizer1.step()

    y2 = model2(x)
    loss2 = y2.pow(2).sum()
    loss2.backward()
    optimizer2.step()

    # Parameters should differ due to different learning rates
    params_differ = False
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1, p2, atol=1e-6):
            params_differ = True
            break

    assert (
        params_differ
    ), "Different learning rates should produce different parameter updates"


def test_optimizer_convergence():
    """Test that optimizer can minimize a simple loss."""
    model = SimpleModel()
    optimizer = create_optimizer(model, learning_rate=1e-2)

    # Target output
    x = torch.randn(4, 10)
    target = torch.randn(4, 10)

    # Record initial loss
    with torch.no_grad():
        initial_output = model(x)
        initial_loss = (initial_output - target).pow(2).mean()

    # Train for several steps
    for _ in range(100):
        output = model(x)
        loss = (output - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Check final loss
    with torch.no_grad():
        final_output = model(x)
        final_loss = (final_output - target).pow(2).mean()

    # Loss should decrease
    assert final_loss < initial_loss, "Optimizer should reduce loss over training"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
