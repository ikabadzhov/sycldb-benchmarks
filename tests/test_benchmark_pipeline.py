import math
import unittest

from scripts import bench_all, plot_measured


class BenchmarkPipelineTests(unittest.TestCase):
    def test_parse_benchmark_output_extracts_all_run_times(self):
        stdout = """
Device: Mock GPU
Run 0: 10.0 ms
Run 1: 12.0 ms
Run 2: 14.0 ms
Avg: 12.0 ms, StdDev: 1.63299 ms
Final result: 42
"""
        self.assertEqual(bench_all.parse_benchmark_output(stdout), [10.0, 12.0, 14.0])

    def test_summarize_times_computes_stats_from_raw_samples(self):
        avg, stddev = plot_measured.summarize_times([10.0, 12.0, 14.0])
        self.assertAlmostEqual(avg, 12.0)
        self.assertTrue(math.isclose(stddev, math.sqrt(8.0 / 3.0), rel_tol=1e-9))

    def test_summarize_times_handles_empty_input(self):
        avg, stddev = plot_measured.summarize_times([])
        self.assertEqual(avg, 0.0)
        self.assertEqual(stddev, 0.0)


if __name__ == "__main__":
    unittest.main()
