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
			logRows.Add("epoch,train_loss,train_accuracy,val_loss,val_accuracy,seconds");

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

				(double valLoss, double valAccuracy) = EvaluateLossAndAccuracy(
					network,
					split.TestInputs,
					validationIndices,
					split.TestLabels);

				double trainLoss = totalTrainLoss / trainIndices.Length;
				double trainAccuracy = 100.0 * trainCorrect / trainIndices.Length;
				double valAccuracyPercent = valAccuracy * 100.0;

				if (valLoss < bestValidationLoss) {
					bestValidationLoss = valLoss;
					bestValidationAccuracy = valAccuracyPercent;
					bestEpoch = epoch;
					network.SaveWeights(Path.Combine(config.WeightDirectory, "best.txt"));
				}

				double seconds = (DateTime.Now - epochStart).TotalSeconds;
				logRows.Add(FormattableString.Invariant(
					$"{epoch},{trainLoss:F6},{trainAccuracy:F3},{valLoss:F6},{valAccuracyPercent:F3},{seconds:F2}"));

				if (epoch == 1 || epoch % logInterval == 0 || epoch == config.Epochs) {
					Console.WriteLine(FormattableString.Invariant(
						$"epoch {epoch,4}  loss {trainLoss:F5}  train {trainAccuracy,6:F2}%  val_loss {valLoss:F5}  val {valAccuracyPercent,6:F2}%  {seconds,5:F2}s"));
				}
			}

			network.SaveWeights(Path.Combine(config.WeightDirectory, "last.txt"));

			double totalSeconds = (DateTime.Now - trainingStart).TotalSeconds;
			string trainLogPath = Path.Combine(config.OutputDirectory, $"train-{config.RunName}.csv");
			File.WriteAllLines(trainLogPath, logRows);

			Console.WriteLine(FormattableString.Invariant(
				$"best epoch {bestEpoch}  val_loss {bestValidationLoss:F5}  val {bestValidationAccuracy:F2}%  total {totalSeconds:F1}s"));
			Console.WriteLine(FormattableString.Invariant($"wrote {trainLogPath} and {Path.Combine(config.WeightDirectory, "best.txt")}"));
		}

		private static double TrainOnSample(NeuralNetwork network, double[] features, int label, double sampleWeight) {
			SetInputs(network, features);
			network.SetDesiredOutput(0, label == 1 ? PositiveTarget : NegativeTarget);
			return network.TrainStep(sampleWeight);
		}

		private static (double loss, double accuracy) EvaluateLossAndAccuracy(
			NeuralNetwork network,
			double[][] inputs,
			int[] indices,
			int[] labels) {
			double totalLoss = 0.0;
			int correct = 0;

			for (int position = 0; position < indices.Length; position++) {
				int index = indices[position];
				SetInputs(network, inputs[index]);
				network.SetDesiredOutput(0, labels[index] == 1 ? PositiveTarget : NegativeTarget);
				network.ForwardPass();

				totalLoss += network.CalculateLoss();
				if (PredictBinary(network) == labels[index]) {
					correct++;
				}
			}

			double avgLoss = indices.Length == 0 ? 0.0 : totalLoss / indices.Length;
			double accuracy = indices.Length == 0 ? 0.0 : (double)correct / indices.Length;
			return (avgLoss, accuracy);
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
	}
}
