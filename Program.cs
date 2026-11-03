using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using DepositPropensityNet.ETL;
using DepositPropensityNet.ML;

namespace DepositPropensityNet {
	internal static class Program {
		private const int InputSize = Loader.FeatureCount;
		private const int OutputCount = 1;
		private const double PositiveTarget = 1.0;
		private const double NegativeTarget = 0.0;
		private const double DecisionThreshold = 0.5;

		private sealed class EvaluationResult {
			public double Loss { get; }
			public ClassificationMetrics Metrics { get; }

			public EvaluationResult(double loss, ClassificationMetrics metrics) {
				Loss = loss;
				Metrics = metrics;
			}
		}

		private static void Main(string[] args) {
			TrainingConfig config;
			try {
				config = TrainingConfig.Parse(args);
			}
			catch (ArgumentException exception) {
				Console.Error.WriteLine(exception.Message);
				Console.Error.WriteLine();
				Console.Error.WriteLine(TrainingConfig.Usage);
				Environment.ExitCode = 1;
				return;
			}

			if (config.ShowHelp) {
				Console.WriteLine(TrainingConfig.Usage);
				return;
			}

			Run(config);
		}

		private static void Run(TrainingConfig config) {
			Console.WriteLine($"Loading dataset '{config.DataPath}'...");
			(double[][] allInputs, int[] allLabels) = Loader.LoadCsv(config.DataPath);

			Loader.DatasetSplit split = Loader.SplitTrainTest(
				allInputs,
				allLabels,
				config.TrainRatio,
				config.Seed);

			Normalizer normalizer = new Normalizer(Loader.ContinuousFeatureCount);
			normalizer.FitTransformInPlace(split.TrainInputs);
			normalizer.TransformInPlace(split.TestInputs);

			int trainCount = Math.Min(config.TrainLimit ?? split.TrainInputs.Length, split.TrainInputs.Length);
			int validationCount = Math.Min(config.ValidationLimit ?? split.TestInputs.Length, split.TestInputs.Length);

			int[] trainIndices = CreateIndices(trainCount);
			int[] validationIndices = CreateIndices(validationCount);

			Directory.CreateDirectory(config.OutputDirectory);
			Directory.CreateDirectory(config.WeightDirectory);
			normalizer.Save(Path.Combine(config.WeightDirectory, "normalizer.txt"));

			Console.WriteLine(FormattableString.Invariant(
				$"config: hidden={config.HiddenCount} lr={config.OutputLearningRate} hidden-lr={config.HiddenLearningRate} pos-weight={config.PositiveClassWeight} epochs={config.Epochs} log-interval={config.EffectiveLogInterval} seed={config.Seed}"));
			Console.WriteLine(FormattableString.Invariant(
				$"split: train={trainIndices.Length} test/val={validationIndices.Length} features={InputSize}"));

			NeuralNetwork network = new NeuralNetwork(
				InputSize,
				config.HiddenCount,
				OutputCount,
				config.OutputLearningRate,
				config.HiddenLearningRate,
				config.Seed);
			Random epochRandom = new Random(config.Seed + 1);

			List<string> logRows = new List<string>(config.Epochs + 1);
			logRows.Add("epoch,train_loss,train_accuracy,val_loss,val_accuracy,val_precision,val_recall,val_f1,seconds");

			double[] trainLosses = new double[config.Epochs];
			double[] validationLosses = new double[config.Epochs];
			double[] trainAccuracies = new double[config.Epochs];
			double[] validationAccuracies = new double[config.Epochs];

			double bestValidationLoss = double.MaxValue;
			double bestValidationAccuracy = 0.0;
			int bestEpoch = 0;
			int logInterval = config.EffectiveLogInterval;
			DateTime trainingStart = DateTime.Now;

			for (int epoch = 1; epoch <= config.Epochs; epoch++) {
				DateTime epochStart = DateTime.Now;
				Shuffle(trainIndices, epochRandom);

				double totalTrainLoss = 0.0;
				int trainCorrect = 0;

				for (int position = 0; position < trainIndices.Length; position++) {
					int index = trainIndices[position];
					int label = split.TrainLabels[index];
					double sampleWeight = label == 1 ? config.PositiveClassWeight : 1.0;

					totalTrainLoss += TrainOnSample(network, split.TrainInputs[index], label, sampleWeight);
					if (PredictBinary(network) == label) {
						trainCorrect++;
					}
				}

				EvaluationResult validation = Evaluate(
					network,
					split.TestInputs,
					validationIndices,
					split.TestLabels);

				double trainLoss = totalTrainLoss / trainIndices.Length;
				double trainAccuracy = 100.0 * trainCorrect / trainIndices.Length;
				double valAccuracyPercent = validation.Metrics.Accuracy * 100.0;

				trainLosses[epoch - 1] = trainLoss;
				validationLosses[epoch - 1] = validation.Loss;
				trainAccuracies[epoch - 1] = trainAccuracy;
				validationAccuracies[epoch - 1] = valAccuracyPercent;

				if (validation.Loss < bestValidationLoss) {
					bestValidationLoss = validation.Loss;
					bestValidationAccuracy = valAccuracyPercent;
					bestEpoch = epoch;
					network.SaveWeights(Path.Combine(config.WeightDirectory, "best.txt"));
				}

				double seconds = (DateTime.Now - epochStart).TotalSeconds;
				logRows.Add(FormattableString.Invariant(
					$"{epoch},{trainLoss:F6},{trainAccuracy:F3},{validation.Loss:F6},{valAccuracyPercent:F3},{validation.Metrics.PositivePrecision:F4},{validation.Metrics.PositiveRecall:F4},{validation.Metrics.PositiveF1:F4},{seconds:F2}"));

				if (epoch == 1 || epoch % logInterval == 0 || epoch == config.Epochs) {
					Console.WriteLine(FormattableString.Invariant(
						$"epoch {epoch,4}  loss {trainLoss:F5}  train {trainAccuracy,6:F2}%  val_loss {validation.Loss:F5}  val {valAccuracyPercent,6:F2}%  prec {validation.Metrics.PositivePrecision * 100.0,5:F1}%  rec {validation.Metrics.PositiveRecall * 100.0,5:F1}%  f1 {validation.Metrics.PositiveF1:F3}  {seconds,5:F2}s"));
				}
			}

			network.SaveWeights(Path.Combine(config.WeightDirectory, "last.txt"));

			double totalSeconds = (DateTime.Now - trainingStart).TotalSeconds;
			string trainLogPath = Path.Combine(config.OutputDirectory, $"train-{config.RunName}.csv");
			File.WriteAllLines(trainLogPath, logRows);

			ChartWriter.WriteLineChart(
				Path.Combine(config.OutputDirectory, $"loss-{config.RunName}.svg"),
				"Training and validation loss",
				"MSE loss",
				new[] { "train", "validation" },
				new[] { trainLosses, validationLosses },
				0.0,
				false);
			ChartWriter.WriteLineChart(
				Path.Combine(config.OutputDirectory, $"accuracy-{config.RunName}.svg"),
				"Training and validation accuracy",
				"accuracy",
				new[] { "train", "validation" },
				new[] { trainAccuracies, validationAccuracies },
				100.0,
				true);

			Console.WriteLine(FormattableString.Invariant(
				$"best epoch {bestEpoch}  val_loss {bestValidationLoss:F5}  val {bestValidationAccuracy:F2}%  total {totalSeconds:F1}s"));
			Console.WriteLine(FormattableString.Invariant($"wrote {trainLogPath}"));

			// Load the best checkpoint for final evaluation and summary reporting.
			NeuralNetwork bestNetwork = new NeuralNetwork(
				InputSize,
				config.HiddenCount,
				OutputCount,
				config.OutputLearningRate,
				config.HiddenLearningRate,
				config.Seed);
			bestNetwork.LoadWeights(Path.Combine(config.WeightDirectory, "best.txt"));

			EvaluationResult bestEval = Evaluate(bestNetwork, split.TestInputs, validationIndices, split.TestLabels);

			if (!config.SkipTest) {
				Console.WriteLine();
				Console.WriteLine(FormattableString.Invariant($"test loss {bestEval.Loss:F5} (at best epoch {bestEpoch})"));
				Console.Write(bestEval.Metrics.FormatTable());

				string testPrefix = Path.Combine(config.OutputDirectory, $"test-{config.RunName}");
				File.WriteAllText(testPrefix + "-metrics.csv", bestEval.Metrics.ToMetricsCsv());
				File.WriteAllText(testPrefix + "-confusion.csv", bestEval.Metrics.ToConfusionCsv());
				ChartWriter.WriteConfusionMatrix(
					testPrefix + "-confusion.svg",
					bestEval.Metrics.GetConfusionMatrix(),
					ClassificationMetrics.ClassNames);
				Console.WriteLine(FormattableString.Invariant($"wrote {testPrefix}-metrics.csv and {testPrefix}-confusion.svg"));
			}

			if (!string.IsNullOrEmpty(config.SummaryCsvPath)) {
				AppendSummary(config, trainIndices.Length, validationIndices.Length, bestEpoch, bestEval, totalSeconds);
			}
		}

		private static double TrainOnSample(NeuralNetwork network, double[] features, int label, double sampleWeight) {
			SetInputs(network, features);
			network.SetDesiredOutput(0, label == 1 ? PositiveTarget : NegativeTarget);
			return network.TrainStep(sampleWeight);
		}

		private static EvaluationResult Evaluate(
			NeuralNetwork network,
			double[][] inputs,
			int[] indices,
			int[] labels) {
			double totalLoss = 0.0;
			int[] actual = new int[indices.Length];
			int[] predicted = new int[indices.Length];

			for (int position = 0; position < indices.Length; position++) {
				int index = indices[position];
				SetInputs(network, inputs[index]);
				network.SetDesiredOutput(0, labels[index] == 1 ? PositiveTarget : NegativeTarget);
				network.ForwardPass();

				totalLoss += network.CalculateLoss();
				actual[position] = labels[index];
				predicted[position] = PredictBinary(network);
			}

			ClassificationMetrics metrics = ClassificationMetrics.Create(actual, predicted);
			double avgLoss = indices.Length == 0 ? 0.0 : totalLoss / indices.Length;
			return new EvaluationResult(avgLoss, metrics);
		}

		private static void SetInputs(NeuralNetwork network, double[] features) {
			for (int inputIndex = 0; inputIndex < features.Length; inputIndex++) {
				network.SetInput(inputIndex, features[inputIndex]);
			}
		}

		private static int PredictBinary(NeuralNetwork network) {
			return network.GetOutput(0) >= DecisionThreshold ? 1 : 0;
		}

		private static int[] CreateIndices(int count) {
			int[] indices = new int[count];
			for (int index = 0; index < count; index++) {
				indices[index] = index;
			}
			return indices;
		}

		private static void Shuffle(int[] indices, Random random) {
			for (int i = indices.Length - 1; i > 0; i--) {
				int j = random.Next(i + 1);
				(indices[i], indices[j]) = (indices[j], indices[i]);
			}
		}

		private static void AppendSummary(
			TrainingConfig config,
			int trainSamples,
			int valSamples,
			int bestEpoch,
			EvaluationResult bestEval,
			double totalSeconds) {
			string? directory = Path.GetDirectoryName(config.SummaryCsvPath);
			if (!string.IsNullOrEmpty(directory)) {
				Directory.CreateDirectory(directory);
			}

			bool writeHeader = !File.Exists(config.SummaryCsvPath);
			using (StreamWriter writer = new StreamWriter(config.SummaryCsvPath!, append: true)) {
				if (writeHeader) {
					writer.WriteLine("data,hidden,output_lr,hidden_lr,pos_weight,epochs,seed,train_samples,val_samples,best_epoch,best_val_loss,accuracy,yes_precision,yes_recall,yes_f1,macro_f1,tn,fp,fn,tp,total_seconds");
				}

				ClassificationMetrics m = bestEval.Metrics;
				writer.WriteLine(FormattableString.Invariant(
					$"{Path.GetFileName(config.DataPath)},{config.HiddenCount},{config.OutputLearningRate:F4},{config.HiddenLearningRate:F4},{config.PositiveClassWeight:F2},{config.Epochs},{config.Seed},{trainSamples},{valSamples},{bestEpoch},{bestEval.Loss:F6},{m.Accuracy * 100.0:F3},{m.PositivePrecision * 100.0:F3},{m.PositiveRecall * 100.0:F3},{m.PositiveF1:F4},{m.MacroF1:F4},{m.TrueNegatives},{m.FalsePositives},{m.FalseNegatives},{m.TruePositives},{totalSeconds:F2}"));
			}
		}
	}
}
