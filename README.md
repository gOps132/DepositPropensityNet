# DepositPropensityNet

A Multi-Layer Perceptron (MLP) binary classifier built from scratch in C# to predict whether a bank client will subscribe to a term deposit (`"yes"` vs `"no"`) based on direct marketing campaign data. It uses no external machine learning libraries—only the custom [`InputNeuron`](src/ML/InputNeuron.cs), [`HiddenNeuron`](src/ML/HiddenNeuron.cs), [`OutputNeuron`](src/ML/OutputNeuron.cs), and [`NeuralNetwork`](src/ML/NeuralNetwork.cs) classes in this repository.

## Built with

<div align="center">

![.NET 10.0](https://img.shields.io/badge/.NET-10.0-512BD4?style=for-the-badge&logo=dotnet&logoColor=white)
![C#](https://img.shields.io/badge/C%23-239120?style=for-the-badge&logo=c-sharp&logoColor=white)

</div>

## Data & ETL Pipeline

UCI Machine Learning Repository: [UCI Bank Marketing Dataset Page](https://archive.ics.uci.edu/dataset/222/bank+marketing)

- **`data/bank/bank-full.csv`**: **45,211 client records** (`39,922` `"no"` [88.30%] and `5,289` `"yes"` [11.70%]).
- **`data/bank/bank.csv`**: Official 10% subset (**4,521 records**). Not used for the reported results (all tuning and final runs use `bank-full.csv`), but handy for quick smoke tests.

### Preprocessing Steps (`src/ETL/`)
1. **CSV Parsing ([`Loader.cs`](src/ETL/Loader.cs))**: Parses semicolon-delimited (`;`) rows, skips the header, and strips double quotes.
2. **Data Leakage Prevention**: Drops column 12 (`duration`, call length in seconds) because call duration is only known *after* a call finishes and leaks the target outcome.
3. **Categorical & Numerical Encoding (47 Input Features)**:
   - **6 Continuous Numerical Features (`[0..5]`)**: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`.
   - **3 Binary Features (`[6..8]`)**: `default`, `housing`, `loan` mapped to `1.0` (`"yes"`) and `0.0` (`"no"`).
   - **6 One-Hot Categorical Features (`[9..46]`, 38 columns)**: `job` (12), `marital` (3), `education` (4), `contact` (3), `month` (12), `poutcome` (4). Explicit `"unknown"` tokens are preserved as their own category so zero rows are lost.
   - **Binary Target (`y`)**: `"yes"` $\to 1$ (`1.0`), `"no"` $\to 0$ (`0.0`).
4. **80/20 Train/Test Split**: Deterministic Fisher-Yates shuffle (`seed = 1234`) splitting `bank-full.csv` into **36,168 training samples (80%)** and **9,043 testing/validation samples (20%)**.
5. **Leakage-Free MinMax Scaling ([`Normalizer.cs`](src/ETL/Normalizer.cs))**: Fits feature minimums and maximums strictly on the 80% training split, transforms continuous features to $[0, 1]$, clamps unseen test values to $[0, 1]$, and saves the scaler bounds to `weights/normalizer.txt`.

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

## Run the trainer

From the project root folder:

```bash
dotnet run -c Release
```

To reproduce the final model on `bank-full.csv` (100 epochs, saving weights, metrics, and SVG charts):

```bash
dotnet run -c Release -- --data data/bank/bank-full.csv --hidden 32 --lr 0.05 --pos-weight 3.0 --epochs 100 --seed 1234
```

All CLI options (`dotnet run -- --help`):

| Option | Default | Meaning |
| --- | --- | --- |
| `--data <path>` | `data/bank/bank-full.csv` | Path to Bank Marketing CSV (`bank-full.csv` or `bank.csv`) |
| `--hidden <n>` | `32` | Number of hidden-layer neurons |
| `--lr <x>` | `0.05` | Output-layer learning rate ($\eta$) |
| `--hidden-lr <x>` | `0.05` | Hidden-layer learning rate |
| `--pos-weight <x>` | `1.0` | Positive class (`"yes"`) error weight for class imbalance |
| `--epochs <n>` | `100` | Number of training passes |
| `--patience <n>` | `0` (off) | Stop after `n` epochs without validation-loss improvement |
| `--log-interval <n>` | `100` | Print console loss every $n$ epochs |
| `--seed <n>` | `1234` | Deterministic seed for split, shuffling, and weight initialization |

## Run the interactive GUI

From the project root folder:

```bash
dotnet run --project src/GUI
```

The GUI is a separate Avalonia project in [`src/GUI/`](src/GUI/), so it uses `--project src/GUI`. It loads `weights/best.txt` and `weights/normalizer.txt` (run the trainer first), lets you enter a bank client's attributes, and press **Predict** to see the predicted outcome (`YES` / `NO`) and propensity score.

## Tuning

All tuning runs use the full `bank-full.csv` (45,211 rows, 80/20 split) with fixed epoch counts (`patience = 0`) and are recorded in [`logs/experiments.csv`](logs/experiments.csv) (63 runs).

**Method**
- Each configuration is run with **3 seeds (1234, 42, 7)**. The seed changes the train/test split, the weight initialization, and the shuffle order, so the spread reflects real run-to-run variation. Tables show the **mean across the 3 seeds** (± standard deviation for validation loss).
- Every run executes for the full specified number of epochs without early stopping (`--patience 0`). The reported evaluation metrics come from the checkpoint (`weights/best.txt`) with the lowest validation loss during the run.
- "Validation" here is the 20% test split, because the project uses a two-way split. Selecting settings and reporting results on the same split is optimistic. Treat the differences as indicative rather than as an unbiased estimate.

### 1. Hidden Layer Size (`lr = 0.05`, `pos-weight = 1.0`, `epochs = 100`)

<details>
<summary>Hidden Layer Size</summary>

| Hidden Nodes | Best Val Loss (± sd) | Accuracy | Precision (`yes`) | Recall (`yes`) | F1 (`yes`) | Best Epoch | Time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.04143 ± 0.00079 | 89.63% | 61.2% | 24.4% | 0.349 | 89 | 1s |
| 8 | 0.04060 ± 0.00066 | 89.62% | 61.7% | 22.8% | 0.332 | 79 | 4s |
| 16 | **0.04044 ± 0.00057** | 89.66% | 61.3% | 24.3% | 0.347 | 86 | 7s |
| **32** | 0.04055 ± 0.00049 | 89.60% | 61.1% | 23.8% | 0.342 | 71 | 13s |
| 64 | 0.04057 ± 0.00062 | 89.52% | 60.0% | 23.6% | 0.338 | 79 | 24s |
| 128 | 0.04065 ± 0.00064 | 89.64% | 60.6% | 25.3% | 0.356 | 75 | 47s |

Verdict: Sizes 8–128 are statistically indistinguishable (validation loss differs by ~0.0002, smaller than seed-to-seed variation of ~0.0006). We kept 32 because the learning-rate and weight sweeps were run at 32, and it is ~2–3.5× cheaper than 64 or 128 with no measurable loss in quality.
</details>

### 2. Learning Rate $\eta$ (`hidden = 32`, `pos-weight = 1.0`, `epochs = 100`)

<details>
<summary>Learning Rate</summary>

| Learning Rate | Best Val Loss (± sd) | Accuracy | F1 (`yes`) | Best Epoch |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.04133 ± 0.00063 | 89.71% | 0.311 | 98 |
| 0.02 | 0.04079 ± 0.00056 | 89.71% | 0.321 | 92 |
| **0.05** | 0.04055 ± 0.00049 | 89.60% | 0.342 | 71 |
| 0.10 | **0.04052 ± 0.00056** | 89.56% | **0.346** | 47 |
| 0.20 | 0.04059 ± 0.00052 | 89.56% | 0.337 | 32 |
| 0.30 | 0.04055 ± 0.00055 | 89.61% | 0.337 | 26 |

Verdict: Every rate from 0.05 to 0.30 gives the same validation loss within noise. Rates 0.01 and 0.02 do not converge within 100 epochs (best-epoch val loss 0.04133 and 0.04079, respectively). 0.05 is a stable middle choice.
</details>

### 3. Positive-Class Weight (`hidden = 32`, `lr = 0.05`, `epochs = 100`)

| Pos Weight | Best Val Loss (± sd) | Accuracy | Precision (`yes`) | Recall (`yes`) | F1 (`yes`) | Best Epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | **0.04055 ± 0.00049** | 89.60% | 61.1% | 23.8% | 0.342 | 71 |
| 2.0 | 0.04118 ± 0.00009 | 89.41% | 56.8% | 32.0% | 0.406 | 62 |
| 2.5 | 0.04225 ± 0.00082 | 89.26% | 54.9% | 33.6% | 0.415 | 46 |
| **3.0** | 0.04360 ± 0.00144 | 89.07% | 52.9% | 35.0% | 0.421 | 17 |
| 4.0 | 0.04706 ± 0.00221 | 88.43% | 49.2% | 42.4% | **0.454** | 46 |

- **Positive-class weight (3.0):** This is a precision/recall trade-off. Raising the weight increases yes recall and F1 (recall 23.8% to 35.0%, F1 0.342 to 0.421 versus unweighted) but lowers precision (61.1% to 52.9%). We stopped at 3.0 rather than 4.0 because 4.0 achieves the best F1 (0.454) but at the cost of precision dropping to 49.2% when the model's purpose is lead prioritization, not per-client prediction. This reflects a preference for balanced precision/recall rather than max F1.
- **Epoch count (100):** Fixed 100 epochs; the best checkpoint is selected by validation loss. 30-epoch runs score about the same F1 (0.425 vs 0.421), so the cap is generous rather than critical.

### Training Curves (seed 1234)

Validation loss bottoms out at epoch 45 (`0.04028`) for the unweighted model and epoch 6 (`0.04302`) for `pos-weight = 3.0`; `weights/best.txt` is that checkpoint. Per-epoch numbers are in [`logs/train-h32-lr0.05-hlr0.05-e100-s1234.csv`](logs/train-h32-lr0.05-hlr0.05-e100-s1234.csv) and [`logs/train-h32-lr0.05-hlr0.05-pw3-e100-s1234.csv`](logs/train-h32-lr0.05-hlr0.05-pw3-e100-s1234.csv).

![Training and validation loss (unweighted)](logs/loss-h32-lr0.05-hlr0.05-e100-s1234.svg)

![Training and validation loss (pos-weight 3.0)](logs/loss-h32-lr0.05-hlr0.05-pw3-e100-s1234.svg)

### Test Set Evaluation (best checkpoint, 9,043 samples)

| Configuration | Best Epoch | Accuracy | Precision (`yes`) | Recall (`yes`) | F1 (`yes`) | TN / FP / FN / TP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Unweighted (`pos-weight = 1.0`) | 45 | 89.75% | 62.57% | 22.47% | 0.331 | 7,887 / 137 / 790 / 229 |
| **Chosen (`pos-weight = 3.0`)** | 6 | 89.03% | 52.11% | 32.78% | 0.402 | 7,717 / 307 / 685 / 334 |

3-seed mean (pos-weight 3.0): F1 0.421 ± 0.029, precision 52.9% ± 3.4%, recall 35.0% ± 2.7%.

Per-class metrics for the chosen configuration:

| Class | Precision | Recall | F1 Score | Support |
| --- | ---: | ---: | ---: | ---: |
| `no (0)` | 91.8% | 96.2% | 94.0% | 8,024 |
| `yes (1)` | 52.1% | 32.8% | 40.2% | 1,019 |
| **Macro Avg** | **72.0%** | **64.5%** | **67.1%** | **9,043** |

![Test confusion matrix (pos-weight 3.0)](logs/test-h32-lr0.05-hlr0.05-pw3-e100-s1234-confusion.svg)

**Analysis**
- Only about 11.3% of test clients subscribe, so always predicting `no` scores about 88.7% accuracy with 0% recall. Both models beat that baseline on accuracy only slightly (89.0% to 89.8%), so accuracy is not the informative metric here.
- What matters is the `yes` class. Among clients the weighted model flags as `yes`, 52.1% actually subscribe, about 4.6 times the 11.3% base rate. It finds 32.8% of all real subscribers, versus 22.5% for the unweighted model, at the cost of 170 more false positives (307 vs 137).
- The model uses only pre-call information (`duration` is dropped to avoid leakage), so its ability to rank clients is modest. This is a lead-prioritization tool, not a reliable per-client predictor.

## File Structure

| Path | Purpose |
| --- | --- |
| [`Program.cs`](Program.cs) | CLI entry point, training loop, evaluation, and CSV/SVG logging |
| [`src/ETL/Loader.cs`](src/ETL/Loader.cs) | Semicolon CSV parser, `duration` drop, 47-feature encoder, and 80/20 split |
| [`src/ETL/Normalizer.cs`](src/ETL/Normalizer.cs) | Leakage-free MinMax $[0, 1]$ normalizer for continuous numerical features |
| [`src/ML/InputNeuron.cs`](src/ML/InputNeuron.cs) | Input-layer neuron storing feature value and input-to-hidden weights |
| [`src/ML/HiddenNeuron.cs`](src/ML/HiddenNeuron.cs) | Hidden-layer neuron with sigmoid activation, bias, delta, and output weights |
| [`src/ML/OutputNeuron.cs`](src/ML/OutputNeuron.cs) | Output-layer neuron producing sigmoid subscription probability |
| [`src/ML/NeuralNetwork.cs`](src/ML/NeuralNetwork.cs) | Multi-Layer Perceptron forward pass, MSE loss, backpropagation, and weight persistence |
| [`src/ML/TrainingConfig.cs`](src/ML/TrainingConfig.cs) | Reproducible command-line hyperparameter configuration parser |
| [`src/ML/Metrics.cs`](src/ML/Metrics.cs) | Binary classification Accuracy, Precision, Recall, F1, and 2×2 confusion matrix |
| [`src/ML/ChartWriter.cs`](src/ML/ChartWriter.cs) | Dependency-free SVG line chart and confusion matrix renderer |
| [`src/GUI/`](src/GUI/) | Avalonia predictor app (`MainWindow.cs`, `App.cs`, `Program.cs`) |
| [`logs/`](logs/) | Epoch training CSV logs, [`experiments.csv`](logs/experiments.csv), and SVG charts |
| [`weights/`](weights/) | Saved model checkpoints (`best.txt`, `last.txt`) and fitted `normalizer.txt` |

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
