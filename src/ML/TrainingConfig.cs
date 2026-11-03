using System;
using System.Globalization;

namespace DepositPropensityNet.ML {
	/// <summary>
	/// Command-line configuration for a training run. Every hyperparameter that
	/// affects the result is captured here so experiments are reproducible.
	/// </summary>
	public sealed class TrainingConfig {
		public string DataPath { get; set; } = "data/bank/bank-full.csv";
		public double TrainRatio { get; set; } = 0.80;
		public int HiddenCount { get; set; } = 32;
		public double OutputLearningRate { get; set; } = 0.05;
		public double HiddenLearningRate { get; set; } = 0.05;
		public double PositiveClassWeight { get; set; } = 1.0;
		public int Epochs { get; set; } = 100;
		public int Patience { get; set; }
		public int? LogInterval { get; set; }
		public int Seed { get; set; } = 1234;
		public int? TrainLimit { get; set; }
		public int? ValidationLimit { get; set; }
		public string OutputDirectory { get; set; } = "logs";
		public string WeightDirectory { get; set; } = "weights";
		public string? SummaryCsvPath { get; set; }
		public bool SkipTest { get; set; }
		public bool ShowHelp { get; set; }

		/// <summary>
		/// Effective epoch logging interval (defaults to 100 when Epochs >= 100, or 10 / 1 for shorter runs).
		/// </summary>
		public int EffectiveLogInterval {
			get {
				if (LogInterval.HasValue) {
					return LogInterval.Value;
				}
				if (Epochs >= 100) {
					return 100;
				}
				return 1;
			}
		}

		/// <summary>A short, file-name-safe description of this configuration.</summary>
		public string RunName {
			get {
				string posWeightSuffix = Math.Abs(PositiveClassWeight - 1.0) > 1e-6
					? FormattableString.Invariant($"-pw{PositiveClassWeight:0.##}")
					: string.Empty;
				return FormattableString.Invariant(
					$"h{HiddenCount}-lr{OutputLearningRate:0.###}-hlr{HiddenLearningRate:0.###}{posWeightSuffix}-e{Epochs}{(Patience > 0 ? $"-p{Patience}" : string.Empty)}-s{Seed}");
			}
		}

		public static string Usage {
			get {
				return string.Join(Environment.NewLine, new[] {
					"Usage: dotnet run -- [options]",
					"",
					"Options:",
					"  --data <path>          Path to Bank Marketing CSV (default data/bank/bank-full.csv)",
					"  --hidden <n>           Hidden layer size (default 32)",
					"  --lr <x>               Output-layer learning rate (default 0.05)",
					"  --hidden-lr <x>        Hidden-layer learning rate (default 0.05)",
					"  --pos-weight <x>       Positive class ('yes') error weight (default 1.0)",
					"  --epochs <n>           Maximum training epochs (default 100)",
					"  --patience <n>         Early stop after n epochs without validation-loss improvement (default 0 = off)",
					"  --log-interval <n>     Print progress every n epochs (default 100 when epochs>=100, else 1)",
					"  --seed <n>             Random seed for weights, split, and shuffling (default 1234)",
					"  --train-limit <n>      Use only n training samples (default: all 80% split)",
					"  --val-limit <n>        Use only n test/validation samples (default: all 20% split)",
					"  --output-dir <path>    Directory for logs and charts (default logs)",
					"  --weight-dir <path>    Directory for saved weights and normalizer (default weights)",
					"  --summary-csv <path>   Append a one-row experiment summary to this CSV",
					"  --skip-test            Skip the final test-set report (useful while tuning)",
					"  --help                 Show this message"
				});
			}
		}

		public static TrainingConfig Parse(string[] args) {
			TrainingConfig config = new TrainingConfig();
			bool hiddenLrExplicitlySet = false;

			for (int index = 0; index < args.Length; index++) {
				string argument = args[index];
				string option = argument;
				string? inlineValue = null;

				int equals = argument.IndexOf('=');
				if (equals >= 0) {
					option = argument.Substring(0, equals);
					inlineValue = argument.Substring(equals + 1);
				}

				switch (option) {
					case "-h":
					case "--help":
						config.ShowHelp = true;
						break;
					case "--data":
						config.DataPath = ReadValue(args, ref index, option, inlineValue);
						break;
					case "--hidden":
						config.HiddenCount = ParseInt(ReadValue(args, ref index, option, inlineValue), option);
						break;
					case "--lr":
						config.OutputLearningRate = ParseDouble(ReadValue(args, ref index, option, inlineValue), option);
						if (!hiddenLrExplicitlySet) {
							config.HiddenLearningRate = config.OutputLearningRate;
						}
						break;
					case "--hidden-lr":
						config.HiddenLearningRate = ParseDouble(ReadValue(args, ref index, option, inlineValue), option);
						hiddenLrExplicitlySet = true;
						break;
					case "--pos-weight":
						config.PositiveClassWeight = ParseDouble(ReadValue(args, ref index, option, inlineValue), option);
						break;
					case "--epochs":
						config.Epochs = ParseInt(ReadValue(args, ref index, option, inlineValue), option);
						break;
					case "--patience":
						config.Patience = ParseInt(ReadValue(args, ref index, option, inlineValue), option);
						break;
					case "--log-interval":
						config.LogInterval = ParseInt(ReadValue(args, ref index, option, inlineValue), option);
						break;
					case "--seed":
						config.Seed = ParseInt(ReadValue(args, ref index, option, inlineValue), option, allowNegative: true);
						break;
					case "--train-limit":
						config.TrainLimit = ParseInt(ReadValue(args, ref index, option, inlineValue), option);
						break;
					case "--val-limit":
						config.ValidationLimit = ParseInt(ReadValue(args, ref index, option, inlineValue), option);
						break;
					case "--output-dir":
						config.OutputDirectory = ReadValue(args, ref index, option, inlineValue);
						break;
					case "--weight-dir":
						config.WeightDirectory = ReadValue(args, ref index, option, inlineValue);
						break;
					case "--summary-csv":
						config.SummaryCsvPath = ReadValue(args, ref index, option, inlineValue);
						break;
					case "--skip-test":
						config.SkipTest = true;
						break;
					default:
						throw new ArgumentException($"Unknown option '{option}'.");
				}
			}

			config.Validate();
			return config;
		}

		private void Validate() {
			if (ShowHelp) {
				return;
			}
			if (string.IsNullOrWhiteSpace(DataPath)) throw new ArgumentException("--data cannot be empty.");
			if (HiddenCount <= 0) throw new ArgumentException("--hidden must be greater than zero.");
			if (OutputLearningRate <= 0.0) throw new ArgumentException("--lr must be greater than zero.");
			if (HiddenLearningRate <= 0.0) throw new ArgumentException("--hidden-lr must be greater than zero.");
			if (PositiveClassWeight <= 0.0) throw new ArgumentException("--pos-weight must be greater than zero.");
			if (Epochs <= 0) throw new ArgumentException("--epochs must be greater than zero.");
			if (LogInterval.HasValue && LogInterval.Value <= 0) throw new ArgumentException("--log-interval must be greater than zero.");
			if (TrainLimit.HasValue && TrainLimit.Value <= 0) throw new ArgumentException("--train-limit must be greater than zero.");
			if (ValidationLimit.HasValue && ValidationLimit.Value <= 0) throw new ArgumentException("--val-limit must be greater than zero.");
			if (string.IsNullOrWhiteSpace(OutputDirectory)) throw new ArgumentException("--output-dir cannot be empty.");
			if (string.IsNullOrWhiteSpace(WeightDirectory)) throw new ArgumentException("--weight-dir cannot be empty.");
			if (SummaryCsvPath != null && string.IsNullOrWhiteSpace(SummaryCsvPath)) throw new ArgumentException("--summary-csv cannot be empty.");
		}

		private static string ReadValue(string[] args, ref int index, string option, string? inlineValue) {
			if (inlineValue != null) {
				if (inlineValue.Length == 0) {
					throw new ArgumentException($"Option '{option}' requires a value.");
				}
				return inlineValue;
			}
			if (index + 1 >= args.Length) {
				throw new ArgumentException($"Option '{option}' requires a value.");
			}
			return args[++index];
		}

		private static int ParseInt(string value, string option, bool allowNegative = false) {
			if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int result)) {
				throw new ArgumentException($"Option '{option}' expects an integer, got '{value}'.");
			}
			if (!allowNegative && result < 0) {
				throw new ArgumentException($"Option '{option}' cannot be negative.");
			}
			return result;
		}

		private static double ParseDouble(string value, string option) {
			if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double result)) {
				throw new ArgumentException($"Option '{option}' expects a number, got '{value}'.");
			}
			return result;
		}
	}
}
