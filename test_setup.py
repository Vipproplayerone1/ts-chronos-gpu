"""Test script to verify environment setup and dependencies."""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")

    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('seaborn', 'seaborn'),
        ('sklearn', 'scikit-learn'),
        ('statsmodels', 'statsmodels'),
        ('lightgbm', 'lightgbm'),
        ('requests', 'requests'),
        ('yaml', 'pyyaml'),
        ('tqdm', 'tqdm'),
    ]

    failed = []
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"  [OK] {package_name}")
        except ImportError as e:
            print(f"  [FAIL] {package_name}: {e}")
            failed.append(package_name)

    # Test PyTorch
    try:
        import torch
        print(f"  [OK] pytorch (version: {torch.__version__})")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  [FAIL] pytorch: {e}")
        failed.append('pytorch')

    # Test Chronos
    try:
        from chronos import ChronosPipeline
        print(f"  [OK] chronos-forecasting")
    except ImportError as e:
        print(f"  [FAIL] chronos-forecasting: {e}")
        failed.append('chronos-forecasting')

    return failed

def test_src_modules():
    """Test if src modules can be imported."""
    print("\nTesting src modules...")

    sys.path.insert(0, '.')

    modules = [
        'src.config',
        'src.data_loader',
        'src.preprocess',
        'src.features',
        'src.metrics',
        'src.backtesting',
        'src.baselines',
        'src.chronos_model',
        'src.stats_tests',
        'src.plots',
        'src.utils'
    ]

    failed = []
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  [OK] {module_name}")
        except Exception as e:
            print(f"  [FAIL] {module_name}: {e}")
            failed.append(module_name)

    return failed

def test_config():
    """Test if configuration file can be loaded."""
    print("\nTesting configuration...")

    try:
        from src.config import load_config
        config = load_config("configs/default.yaml")
        print(f"  [OK] Configuration loaded")
        print(f"    Random seed: {config.random_seed}")
        print(f"    Page title: {config.page_title}")
        print(f"    Horizon: {config.horizon}")
        return True
    except Exception as e:
        print(f"  [FAIL] Configuration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("  Setup Verification Test")
    print("="*60)

    # Test imports
    failed_packages = test_imports()

    # Test src modules
    failed_modules = test_src_modules()

    # Test config
    config_ok = test_config()

    # Summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)

    if failed_packages:
        print(f"\n[X] Missing packages: {', '.join(failed_packages)}")
        print("   Run: pip install " + " ".join(failed_packages))
    else:
        print("\n[OK] All packages installed")

    if failed_modules:
        print(f"\n[X] Failed src modules: {', '.join(failed_modules)}")
    else:
        print("[OK] All src modules loadable")

    if not config_ok:
        print("\n[X] Configuration loading failed")
    else:
        print("[OK] Configuration OK")

    if not failed_packages and not failed_modules and config_ok:
        print("\n" + "="*60)
        print("  [SUCCESS] ALL TESTS PASSED - Ready to run pipeline!")
        print("="*60)
        print("\nRun the pipeline with:")
        print("  python run_pipeline.py")
        return 0
    else:
        print("\n" + "="*60)
        print("  [FAILED] TESTS FAILED - Fix issues above")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
