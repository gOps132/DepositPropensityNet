using System;
using System.Globalization;
using System.IO;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using DepositPropensityNet.ETL;
using DepositPropensityNet.ML;

namespace DepositPropensityNet.GUI {
	/// <summary>
	/// Simple interactive predictor window that loads weights/best.txt and weights/normalizer.txt,
	/// lets the user enter a bank client's attributes, and predicts whether the client will
	/// subscribe to a term deposit ("yes" vs "no").
	/// </summary>
	public sealed class MainWindow : Window {
		private const int InputCount = Loader.FeatureCount;
		private const int OutputCount = 1;

		private readonly TextBox ageInput = new TextBox();
		private readonly ComboBox jobInput = new ComboBox();
		private readonly ComboBox maritalInput = new ComboBox();
		private readonly ComboBox educationInput = new ComboBox();
		private readonly ComboBox defaultInput = new ComboBox();
		private readonly TextBox balanceInput = new TextBox();
		private readonly ComboBox housingInput = new ComboBox();
		private readonly ComboBox loanInput = new ComboBox();
		private readonly ComboBox contactInput = new ComboBox();
		private readonly TextBox dayInput = new TextBox();
		private readonly ComboBox monthInput = new ComboBox();
		private readonly TextBox campaignInput = new TextBox();
		private readonly TextBox pdaysInput = new TextBox();
		private readonly TextBox previousInput = new TextBox();
		private readonly ComboBox poutcomeInput = new ComboBox();

		private readonly TextBlock resultText = new TextBlock();
		private readonly NeuralNetwork network;
		private readonly Normalizer normalizer;

		public MainWindow() {
			Title = "Deposit Propensity Predictor";
			Width = 480;
			Height = 520;
			CanResize = false;

			InitCombo(jobInput, Loader.JobCategories, "management");
			InitCombo(maritalInput, Loader.MaritalCategories, "married");
			InitCombo(educationInput, Loader.EducationCategories, "tertiary");
			InitCombo(defaultInput, new[] { "no", "yes" }, "no");
			InitCombo(housingInput, new[] { "no", "yes" }, "no");
			InitCombo(loanInput, new[] { "no", "yes" }, "no");
			InitCombo(contactInput, Loader.ContactCategories, "cellular");
			InitCombo(monthInput, Loader.MonthCategories, "may");
			InitCombo(poutcomeInput, Loader.PoutcomeCategories, "unknown");

			ResetDefaults();

			Grid form = new Grid {
				ColumnDefinitions = new ColumnDefinitions("Auto,*,Auto,*"),
				RowDefinitions = new RowDefinitions("Auto,Auto,Auto,Auto,Auto,Auto,Auto,Auto")
			};

			AddField(form, 0, 0, "Age:", ageInput);
			AddField(form, 0, 2, "Job:", jobInput);
			AddField(form, 1, 0, "Marital:", maritalInput);
			AddField(form, 1, 2, "Education:", educationInput);
			AddField(form, 2, 0, "Default:", defaultInput);
			AddField(form, 2, 2, "Balance (€):", balanceInput);
			AddField(form, 3, 0, "Housing:", housingInput);
			AddField(form, 3, 2, "Loan:", loanInput);
			AddField(form, 4, 0, "Contact:", contactInput);
			AddField(form, 4, 2, "Day (1-31):", dayInput);
			AddField(form, 5, 0, "Month:", monthInput);
			AddField(form, 5, 2, "Campaign:", campaignInput);
			AddField(form, 6, 0, "Pdays:", pdaysInput);
			AddField(form, 6, 2, "Previous:", previousInput);
			AddField(form, 7, 0, "Poutcome:", poutcomeInput);

			Button predictButton = new Button {
				Content = "Predict",
				Width = 100
			};
			predictButton.Click += (sender, args) => Predict();

			Button resetButton = new Button {
				Content = "Reset",
				Width = 100
			};
			resetButton.Click += (sender, args) => {
				ResetDefaults();
				resultText.Text = "Enter client details, then press Predict.";
				resultText.Foreground = Brushes.Black;
			};

			StackPanel buttons = new StackPanel {
				Orientation = Orientation.Horizontal,
				Spacing = 10,
				Children = { predictButton, resetButton }
			};

			resultText.FontSize = 16;
			resultText.FontWeight = FontWeight.Bold;
			resultText.TextWrapping = TextWrapping.Wrap;
			resultText.Text = "Enter client details, then press Predict.";

			StackPanel layout = new StackPanel {
				Margin = new Thickness(14),
				Spacing = 14,
				Children = {
					form,
					buttons,
					resultText
				}
			};

			Content = layout;

			string weightsDirectory = ResolveWeightsDirectory();
			string bestWeights = Path.Combine(weightsDirectory, "best.txt");
			string fallbackWeights = Path.Combine(weightsDirectory, "last.txt");
			string weightsPath = File.Exists(bestWeights) ? bestWeights : fallbackWeights;
			string normalizerPath = Path.Combine(weightsDirectory, "normalizer.txt");

			// Infer the hidden-layer size from the checkpoint so the app works
			// with models trained using any --hidden value.
			int valuesPerHiddenNeuron = InputCount + OutputCount + 1;
			int lineCount = File.ReadAllLines(weightsPath).Length;
			int hiddenCount = (lineCount - OutputCount) / valuesPerHiddenNeuron;

			if (hiddenCount <= 0 || hiddenCount * valuesPerHiddenNeuron + OutputCount != lineCount) {
				throw new InvalidDataException($"Unrecognized weights file: {weightsPath}");
			}

			network = new NeuralNetwork(InputCount, hiddenCount, OutputCount, 0.05, 0.05, 1234);
			network.LoadWeights(weightsPath);
			normalizer = Normalizer.Load(normalizerPath);
		}

		private void Predict() {
			try {
				string[] tokens = {
					ageInput.Text?.Trim() ?? "41",
					jobInput.SelectedItem?.ToString() ?? "management",
					maritalInput.SelectedItem?.ToString() ?? "married",
					educationInput.SelectedItem?.ToString() ?? "tertiary",
					defaultInput.SelectedItem?.ToString() ?? "no",
					balanceInput.Text?.Trim() ?? "1500",
					housingInput.SelectedItem?.ToString() ?? "no",
					loanInput.SelectedItem?.ToString() ?? "no",
					contactInput.SelectedItem?.ToString() ?? "cellular",
					dayInput.Text?.Trim() ?? "15",
					monthInput.SelectedItem?.ToString() ?? "may",
					"0", // duration (dropped to prevent data leakage)
					campaignInput.Text?.Trim() ?? "1",
					pdaysInput.Text?.Trim() ?? "-1",
					previousInput.Text?.Trim() ?? "0",
					poutcomeInput.SelectedItem?.ToString() ?? "unknown",
					"no"
				};

				double[] input = Loader.EncodeRow(tokens);
				normalizer.TransformRowInPlace(input);

				for (int inputIndex = 0; inputIndex < input.Length; inputIndex++) {
					network.SetInput(inputIndex, input[inputIndex]);
				}
				network.ForwardPass();

				double probability = network.GetOutput(0);
				bool subscribed = probability >= 0.5;
				string outcome = subscribed ? "YES (Subscribed)" : "NO (Did Not Subscribe)";
				double confidence = subscribed ? probability : (1.0 - probability);

				resultText.Text = FormattableString.Invariant(
					$"Prediction: {outcome}\nPropensity Score: {probability:P1}   (Confidence: {confidence:P1})");
				resultText.Foreground = subscribed
					? new SolidColorBrush(Color.Parse("#15803D"))
					: new SolidColorBrush(Color.Parse("#B91C1C"));
			}
			catch (Exception ex) {
				resultText.Text = $"Invalid input: {ex.Message}";
				resultText.Foreground = new SolidColorBrush(Color.Parse("#B91C1C"));
			}
		}

		private void ResetDefaults() {
			ageInput.Text = "41";
			balanceInput.Text = "1500";
			dayInput.Text = "15";
			campaignInput.Text = "1";
			pdaysInput.Text = "-1";
			previousInput.Text = "0";

			SelectCombo(jobInput, "management");
			SelectCombo(maritalInput, "married");
			SelectCombo(educationInput, "tertiary");
			SelectCombo(defaultInput, "no");
			SelectCombo(housingInput, "no");
			SelectCombo(loanInput, "no");
			SelectCombo(contactInput, "cellular");
			SelectCombo(monthInput, "may");
			SelectCombo(poutcomeInput, "unknown");
		}

		private static string ResolveWeightsDirectory() {
			string baseWeights = Path.Combine(AppContext.BaseDirectory, "weights");
			if (File.Exists(Path.Combine(baseWeights, "best.txt"))) {
				return baseWeights;
			}
			string repoWeights = Path.Combine(Directory.GetCurrentDirectory(), "weights");
			if (File.Exists(Path.Combine(repoWeights, "best.txt"))) {
				return repoWeights;
			}
			return baseWeights;
		}

		private static void InitCombo(ComboBox box, string[] items, string defaultItem) {
			box.ItemsSource = items;
			box.HorizontalAlignment = HorizontalAlignment.Stretch;
			SelectCombo(box, defaultItem);
		}

		private static void SelectCombo(ComboBox box, string item) {
			if (box.ItemsSource is string[] items) {
				int index = Array.IndexOf(items, item);
				box.SelectedIndex = index >= 0 ? index : 0;
			}
		}

		private static void AddField(Grid grid, int row, int col, string labelText, Control control) {
			TextBlock label = new TextBlock {
				Text = labelText,
				VerticalAlignment = VerticalAlignment.Center,
				Margin = new Thickness(4, 6, 6, 6)
			};
			control.Margin = new Thickness(0, 4, 8, 4);
			Grid.SetRow(label, row);
			Grid.SetColumn(label, col);
			Grid.SetRow(control, row);
			Grid.SetColumn(control, col + 1);
			grid.Children.Add(label);
			grid.Children.Add(control);
		}
	}
}
