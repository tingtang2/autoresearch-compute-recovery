"""Environment/package prompt."""

import random


def get_prompt_environment():
    """Installed packages description."""
    pkgs = [
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "xgboost",
        "lightGBM",
        # "torch",
        # "torchvision",
        # "torch-geometric",
        "bayesian-optimization",
        # "timm",
        # "transformers",
        # "sentence-transformers",
        # "opencv-python",
        "Pillow",
    ]
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    return {
        "Installed Packages": f"The following packages are available: {pkg_str}. ⚠️ You MUST focus on classic machine learning approaches and refrain from using neural networks. Only CPU-enabled packages are available."
    }
