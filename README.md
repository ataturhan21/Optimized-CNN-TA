# Optimized-CNN-TA

A notebook-based research project comparing trading strategies built with technical analysis, machine learning, and deep learning.

## Explore the project

| Resource | Contents |
| --- | --- |
| [Main notebook](notebooks/CNN-TA.ipynb) | The experiment workflow |
| [Source modules](src/) | Data preparation, feature engineering, neural networks, strategies, and backtesting |
| [Saved models](models/) | CNN, GRU, LSTM, and MLP model files |
| [Research article](https://drive.google.com/file/d/1s4Az5HqdqpjbkPw00iGHiUh8bYuAJOHp/view?usp=share_link) | The accompanying write-up |

Start with the notebook to inspect the experiments, then follow the source modules for implementation details.

## Environment

[`requirements.txt`](requirements.txt) records the original development environment. It includes a machine-specific Windows path and platform-specific dependencies, so it needs adaptation before installation on another machine.

The bundled `TA_Lib-0.4.24-cp37-cp37m-win_amd64.whl` targets CPython 3.7 on 64-bit Windows. Use an appropriate TA-Lib installation for your environment.

Review the notebook's data and model paths before running its cells. The repository preserves the research implementation; this documentation update does not establish compatibility with current package versions or reproduce the experiment results.

## Feature preprocessing checks

Feature scaling is fitted on the training period. The `Label` column is kept separate from numeric features so a training period missing a class cannot change the meanings of hold, buy, and sell labels.

```bash
python -m unittest discover -s tests -v
```

These checks require NumPy, pandas, and scikit-learn. They verify label preservation, training-only scaling, input immutability, and explicit errors for unsupported scaling methods. Image-window checks also verify label alignment when `Label` is not the final column, trailing-window order, and explicit shape errors. The tests do not run TensorFlow or a backtest.

## License

[MIT](LICENSE).
