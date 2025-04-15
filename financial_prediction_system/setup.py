from setuptools import setup, find_packages

setup(
    name="financial_prediction_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.6.1",
        "scipy>=1.15.2",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "joblib>=1.4.2",
        "threadpoolctl>=3.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0",
        ],
    },
    python_requires=">=3.8",
) 