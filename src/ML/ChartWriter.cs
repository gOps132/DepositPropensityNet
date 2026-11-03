using System;
using System.IO;
using System.Text;

namespace DepositPropensityNet.ML {
	/// <summary>
	/// Writes dependency-free SVG charts: training/validation curves and binary confusion matrices.
	/// </summary>
	public static class ChartWriter {
		private const int Width = 760;
		private const int Height = 430;
		private const int MarginLeft = 75;
		private const int MarginRight = 30;
		private const int MarginTop = 60;
		private const int MarginBottom = 60;

		private static readonly string[] Palette = { "#2563eb", "#ea580c", "#16a34a", "#9333ea" };

		/// <summary>Draws one line per series across epochs.</summary>
		public static void WriteLineChart(
			string filePath,
			string title,
			string yLabel,
			string[] seriesNames,
			double[][] series,
			double yMaximum,
			bool percentage) {
			if (series.Length != seriesNames.Length) {
				throw new ArgumentException("Each series needs a name.");
			}

			int pointCount = series.Length == 0 ? 0 : series[0].Length;
			for (int seriesIndex = 0; seriesIndex < series.Length; seriesIndex++) {
				if (series[seriesIndex].Length != pointCount) {
					throw new ArgumentException("All series must have the same length.");
				}
			}
			if (pointCount == 0) {
				return;
			}

			double maximum;
			if (yMaximum > 0.0) {
				maximum = yMaximum;
			}
			else {
				maximum = 0.0;
				for (int seriesIndex = 0; seriesIndex < series.Length; seriesIndex++) {
					for (int point = 0; point < pointCount; point++) {
						if (series[seriesIndex][point] > maximum) {
							maximum = series[seriesIndex][point];
						}
					}
				}
				if (maximum <= 0.0) {
					maximum = 1.0;
				}
				maximum *= 1.05;
			}

			double plotLeft = MarginLeft;
			double plotTop = MarginTop;
			double plotWidth = Width - MarginLeft - MarginRight;
			double plotHeight = Height - MarginTop - MarginBottom;

			StringBuilder svg = new StringBuilder();
			svg.AppendLine(FormattableString.Invariant(
				$"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{Width}\" height=\"{Height}\" viewBox=\"0 0 {Width} {Height}\">"));
			svg.AppendLine("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>");
			svg.AppendLine(FormattableString.Invariant(
				$"<text x=\"{Width / 2}\" y=\"30\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"18\" font-weight=\"600\" fill=\"#111827\">{Escape(title)}</text>"));

			// Horizontal grid lines and y-axis labels.
			const int gridLines = 5;
			for (int line = 0; line <= gridLines; line++) {
				double fraction = (double)line / gridLines;
				double y = plotTop + plotHeight - fraction * plotHeight;
				double value = maximum * fraction;
				string label = percentage
					? FormattableString.Invariant($"{value:0}%")
					: FormattableString.Invariant($"{value:0.000}");

				svg.AppendLine(FormattableString.Invariant(
					$"<line x1=\"{plotLeft:0.##}\" y1=\"{y:0.##}\" x2=\"{plotLeft + plotWidth:0.##}\" y2=\"{y:0.##}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>"));
				svg.AppendLine(FormattableString.Invariant(
					$"<text x=\"{plotLeft - 12:0.##}\" y=\"{y + 4:0.##}\" text-anchor=\"end\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"12\" fill=\"#4b5563\">{label}</text>"));
			}

			// Axes.
			svg.AppendLine(FormattableString.Invariant(
				$"<line x1=\"{plotLeft:0.##}\" y1=\"{plotTop:0.##}\" x2=\"{plotLeft:0.##}\" y2=\"{plotTop + plotHeight:0.##}\" stroke=\"#9ca3af\" stroke-width=\"1.5\"/>"));
			svg.AppendLine(FormattableString.Invariant(
				$"<line x1=\"{plotLeft:0.##}\" y1=\"{plotTop + plotHeight:0.##}\" x2=\"{plotLeft + plotWidth:0.##}\" y2=\"{plotTop + plotHeight:0.##}\" stroke=\"#9ca3af\" stroke-width=\"1.5\"/>"));

			// X-axis ticks.
			int tickStep = Math.Max(1, (int)Math.Ceiling(pointCount / 10.0));
			for (int point = 0; point < pointCount; point += tickStep) {
				double x = pointCount == 1
					? plotLeft + plotWidth / 2.0
					: plotLeft + plotWidth * point / (pointCount - 1.0);
				svg.AppendLine(FormattableString.Invariant(
					$"<line x1=\"{x:0.##}\" y1=\"{plotTop + plotHeight:0.##}\" x2=\"{x:0.##}\" y2=\"{plotTop + plotHeight + 5:0.##}\" stroke=\"#9ca3af\" stroke-width=\"1\"/>"));
				svg.AppendLine(FormattableString.Invariant(
					$"<text x=\"{x:0.##}\" y=\"{plotTop + plotHeight + 22:0.##}\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"12\" fill=\"#4b5563\">{point + 1}</text>"));
			}

			// Axis labels.
			svg.AppendLine(FormattableString.Invariant(
				$"<text x=\"{plotLeft + plotWidth / 2.0:0.##}\" y=\"{Height - 15}\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"13\" fill=\"#374151\">epoch</text>"));
			svg.AppendLine(FormattableString.Invariant(
				$"<text x=\"20\" y=\"{plotTop + plotHeight / 2.0:0.##}\" text-anchor=\"middle\" transform=\"rotate(-90 20 {plotTop + plotHeight / 2.0:0.##})\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"13\" fill=\"#374151\">{Escape(yLabel)}</text>"));

			// Series.
			bool drawCircles = pointCount <= 40;
			for (int seriesIndex = 0; seriesIndex < series.Length; seriesIndex++) {
				double[] values = series[seriesIndex];
				string color = Palette[seriesIndex % Palette.Length];
				StringBuilder points = new StringBuilder();

				for (int point = 0; point < pointCount; point++) {
					double x = pointCount == 1
						? plotLeft + plotWidth / 2.0
						: plotLeft + plotWidth * point / (pointCount - 1.0);
					double y = plotTop + plotHeight - (values[point] / maximum) * plotHeight;
					points.Append(FormattableString.Invariant($"{x:0.##},{y:0.##} "));
				}

				svg.AppendLine(FormattableString.Invariant(
					$"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2.2\" stroke-linejoin=\"round\" stroke-linecap=\"round\" points=\"{points.ToString().Trim()}\"/>"));

				if (drawCircles) {
					for (int point = 0; point < pointCount; point++) {
						double x = pointCount == 1
							? plotLeft + plotWidth / 2.0
							: plotLeft + plotWidth * point / (pointCount - 1.0);
						double y = plotTop + plotHeight - (values[point] / maximum) * plotHeight;
						svg.AppendLine(FormattableString.Invariant(
							$"<circle cx=\"{x:0.##}\" cy=\"{y:0.##}\" r=\"3\" fill=\"{color}\"/>"));
					}
				}
			}

			// Legend.
			double legendX = plotLeft + plotWidth - 10;
			double legendY = plotTop + 5;
			for (int seriesIndex = 0; seriesIndex < series.Length; seriesIndex++) {
				string color = Palette[seriesIndex % Palette.Length];
				double itemY = legendY + seriesIndex * 20;
				svg.AppendLine(FormattableString.Invariant(
					$"<line x1=\"{legendX - 90:0.##}\" y1=\"{itemY:0.##}\" x2=\"{legendX - 70:0.##}\" y2=\"{itemY:0.##}\" stroke=\"{color}\" stroke-width=\"3\"/>"));
				svg.AppendLine(FormattableString.Invariant(
					$"<text x=\"{legendX - 63:0.##}\" y=\"{itemY + 4:0.##}\" text-anchor=\"start\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"12\" fill=\"#374151\">{Escape(seriesNames[seriesIndex])}</text>"));
			}

			svg.AppendLine("</svg>");
			WriteFile(filePath, svg.ToString());
		}

		/// <summary>Draws a heat-mapped 2x2 binary confusion matrix (actual rows, predicted columns).</summary>
		public static void WriteConfusionMatrix(string filePath, int[,] confusion, string[] classLabels) {
			int classCount = classLabels.Length;
			const int cellSize = 110;
			const int marginLeft = 100;
			const int marginTop = 90;
			const int marginRight = 40;
			const int marginBottom = 70;

			int width = marginLeft + classCount * cellSize + marginRight;
			int height = marginTop + classCount * cellSize + marginBottom;

			int maximum = 1;
			for (int actual = 0; actual < classCount; actual++) {
				for (int predicted = 0; predicted < classCount; predicted++) {
					if (confusion[actual, predicted] > maximum) {
						maximum = confusion[actual, predicted];
					}
				}
			}

			StringBuilder svg = new StringBuilder();
			svg.AppendLine(FormattableString.Invariant(
				$"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"));
			svg.AppendLine("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>");
			svg.AppendLine(FormattableString.Invariant(
				$"<text x=\"{width / 2}\" y=\"30\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"18\" font-weight=\"600\" fill=\"#111827\">Confusion Matrix</text>"));

			for (int classIndex = 0; classIndex < classCount; classIndex++) {
				string label = Escape(classLabels[classIndex]);
				svg.AppendLine(FormattableString.Invariant(
					$"<text x=\"{marginLeft + classIndex * cellSize + cellSize / 2.0:0.##}\" y=\"{marginTop - 12}\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"13\" font-weight=\"600\" fill=\"#374151\">{label}</text>"));
				svg.AppendLine(FormattableString.Invariant(
					$"<text x=\"{marginLeft - 14}\" y=\"{marginTop + classIndex * cellSize + cellSize / 2.0 + 4:0.##}\" text-anchor=\"end\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"13\" font-weight=\"600\" fill=\"#374151\">{label}</text>"));
			}

			svg.AppendLine(FormattableString.Invariant(
				$"<text x=\"{marginLeft + classCount * cellSize / 2.0:0.##}\" y=\"{marginTop - 38}\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"13\" fill=\"#4b5563\">predicted</text>"));
			svg.AppendLine(FormattableString.Invariant(
				$"<text x=\"24\" y=\"{marginTop + classCount * cellSize / 2.0:0.##}\" text-anchor=\"middle\" transform=\"rotate(-90 24 {marginTop + classCount * cellSize / 2.0:0.##})\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"13\" fill=\"#4b5563\">actual</text>"));

			for (int actual = 0; actual < classCount; actual++) {
				for (int predicted = 0; predicted < classCount; predicted++) {
					int count = confusion[actual, predicted];
					double intensity = (double)count / maximum;
					double opacity = 0.10 + 0.80 * intensity;
					string textColor = intensity > 0.55 ? "#ffffff" : "#111827";
					double x = marginLeft + predicted * cellSize;
					double y = marginTop + actual * cellSize;

					svg.AppendLine(FormattableString.Invariant(
						$"<rect x=\"{x:0.##}\" y=\"{y:0.##}\" width=\"{cellSize - 4}\" height=\"{cellSize - 4}\" rx=\"6\" fill=\"#2563eb\" fill-opacity=\"{opacity:0.###}\"/>"));
					svg.AppendLine(FormattableString.Invariant(
						$"<text x=\"{x + cellSize / 2.0 - 2:0.##}\" y=\"{y + cellSize / 2.0 + 5:0.##}\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"15\" font-weight=\"600\" fill=\"{textColor}\">{count}</text>"));
				}
			}

			svg.AppendLine(FormattableString.Invariant(
				$"<text x=\"{marginLeft}\" y=\"{marginTop + classCount * cellSize + 32}\" font-family=\"Segoe UI, Arial, sans-serif\" font-size=\"12\" fill=\"#4b5563\">Max cell count: {maximum}</text>"));
			svg.AppendLine("</svg>");
			WriteFile(filePath, svg.ToString());
		}

		private static void WriteFile(string filePath, string contents) {
			string? directory = Path.GetDirectoryName(filePath);
			if (!string.IsNullOrEmpty(directory)) {
				Directory.CreateDirectory(directory);
			}
			File.WriteAllText(filePath, contents);
		}

		private static string Escape(string value) {
			return value
				.Replace("&", "&amp;")
				.Replace("<", "&lt;")
				.Replace(">", "&gt;")
				.Replace("\"", "&quot;");
		}
	}
}
