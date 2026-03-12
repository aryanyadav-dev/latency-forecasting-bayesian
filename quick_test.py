#!/usr/bin/env python3
"""Quick validation script to verify all modules can be imported and basic functionality works."""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    modules = [
        'data.tokenizer',
        'data.dataset_loader',
        'models.encoder',
        'models.forecasting_network',
        'models.decoder',
        'models.complete_model',
        'models.baseline_models',
        'training.loss_functions',
        'training.optimizer',
        'training.scheduler',
        'training.trainer',
        'evaluation.metrics',
        'evaluation.latent_analysis',
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failed.append((module, e))
    
    return len(failed) == 0, failed

def test_basic_functionality():
    """Test basic functionality without heavy computation."""
    print("\nTesting basic functionality...")
    
    try:
        # Test model creation
        from models.encoder import Encoder
        encoder = Encoder(vocab_size=100, latent_dim=64, num_layers=2, num_heads=4, hidden_dim=256, dropout=0.1)
        print("  ✓ Encoder creation")
        
        # Test loss functions
        import torch
        from training.loss_functions import compute_token_loss
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        loss = compute_token_loss(logits, labels)
        assert torch.isfinite(loss), "Loss is not finite"
        print("  ✓ Loss computation")
        
        # Test metrics
        from evaluation.metrics import compute_perplexity
        test_loss = torch.tensor(2.0)
        ppl = compute_perplexity(test_loss)
        assert ppl > 0, "Perplexity must be positive"
        print("  ✓ Metrics computation")
        
        # Test latent analysis
        import numpy as np
        from evaluation.latent_analysis import compute_latent_entropy
        latents = np.random.randn(100, 64)
        entropy = compute_latent_entropy(latents)
        assert entropy >= 0, "Entropy must be non-negative"
        print("  ✓ Latent analysis")
        
        return True, None
    except Exception as e:
        traceback.print_exc()
        return False, e

def main():
    print("=" * 60)
    print("Quick Validation Test Suite")
    print("=" * 60)
    
    # Test imports
    imports_ok, import_failures = test_imports()
    
    # Test basic functionality
    if imports_ok:
        func_ok, func_error = test_basic_functionality()
    else:
        func_ok = False
        func_error = "Skipped due to import failures"
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if imports_ok and func_ok:
        print("✓ All quick tests passed!")
        print("\nNote: Full test suite with dataset downloads skipped for speed.")
        print("Run 'pytest tests/' for comprehensive testing when needed.")
        return 0
    else:
        print("✗ Some tests failed:")
        if not imports_ok:
            print("\nImport failures:")
            for module, error in import_failures:
                print(f"  - {module}: {error}")
        if not func_ok:
            print(f"\nFunctionality test failed: {func_error}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
