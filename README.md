# Optimized-CNN-TA

A notebook-based research project comparing trading strategies built with technical analysis, machine learning, and deep learning.

This is research and educational code, not financial advice.

## Explore the project

| Resource | Contents |
| --- | --- |
| [Main notebook](notebooks/CNN-TA.ipynb) | The experiment workflow |
| [Source modules](src/) | Data preparation, feature engineering, neural networks, strategies, and backtesting |
| [Saved models](models/) | CNN, GRU, LSTM, and MLP model files |
| [Research article](https://drive.google.com/file/d/1s4Az5HqdqpjbkPw00iGHiUh8bYuAJOHp/view?usp=share_link) | The accompanying write-up |

Start with the notebook to inspect the experiments, then follow the source modules for implementation details.

## Environment

[`requirements.txt`](requirements.txt) records the original development environment. It includes platform-specific dependencies (for example `tensorflow-intel` and `pywinpty` for Windows), so it needs adaptation before installation on another machine.

The bundled `TA_Lib-0.4.24-cp37-cp37m-win_amd64.whl` targets CPython 3.7 on 64-bit Windows. Use an appropriate TA-Lib installation for your environment.

Review the notebook's data and model paths before running its cells. The repository preserves the research implementation; this documentation update does not establish compatibility with current package versions or reproduce the experiment results.

## Offline checks

Feature scaling is fitted on the training period. The `Label` column is kept separate from numeric features so a training period missing a class cannot change the meanings of hold, buy, and sell labels.

CI runs these checks with Python 3.12 and only NumPy, pandas, and scikit-learn:

```bash
python -m pip install numpy==2.3.5 pandas==2.2.3 scikit-learn==1.8.0
python -m unittest discover -s tests -v
```

They verify label preservation, training-only scaling, input immutability, and explicit errors for unsupported scaling and feature-selection methods. Image-window checks also verify label alignment when `Label` is not the final column, trailing-window order, and explicit shape errors. Further checks run the support/resistance clustering on synthetic prices and the FRED forward fill with a fake reader. The tests do not download data, run TensorFlow, or run a backtest.

## License

[MIT](LICENSE).
