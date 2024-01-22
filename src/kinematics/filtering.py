import math

import numpy as np


class ButterworthFilter:
    def __init__(self):
        self.a0, self.a1, self.a2 = 0, 0, 0
        self.b1, self.b2 = 0, 0
        self.x0, self.x1, self.x2 = 0, 0, 0
        self.y0, self.y1, self.y2 = 0, 0, 0
        self.correction_factor = 0

    def filter_samples(self, samples, fs, fc_tests):
        self.update_correction_factor(2)
        padding = 10
        padded = self.add_padding(samples, padding)

        filtered_results = []
        best_cutoff_index = -1

        for fc in np.linspace(0.5, fs / 2, fc_tests):
            filtered_padded = self.filter_samples_internal(padded, fs, fc)
            filtered = self.remove_padding(filtered_padded, padding)
            residuals = np.array(samples) - filtered
            dw = self.durbin_watson(residuals)

            if dw is not None and not math.isnan(dw):
                dw_normalized = abs(2 - dw) / 2
                result = FilteringResult(fc, filtered, dw_normalized)
                filtered_results.append(result)

                if (
                    best_cutoff_index == -1
                    or dw_normalized < filtered_results[best_cutoff_index].dw_normalized
                ):
                    best_cutoff_index = len(filtered_results) - 1

        return filtered_results, best_cutoff_index

    def update_correction_factor(self, passes):
        self.correction_factor = math.pow((math.pow(2, 1.0 / passes) - 1), 0.25)

    def update_coefficients(self, fs, fc):
        o = math.tan(math.pi * fc / fs) / self.correction_factor
        k1 = math.sqrt(2) * o
        k2 = o * o

        self.a0 = k2 / (1 + k1 + k2)
        self.a1 = 2 * self.a0
        self.a2 = self.a0

        k3 = 2 * self.a0 / k2
        self.b1 = -2 * self.a0 + k3
        self.b2 = 1 - self.a0 - self.a1 - self.a2 - self.b1

    def reset_values(self):
        self.x0, self.x1, self.x2 = 0, 0, 0
        self.y0, self.y1, self.y2 = 0, 0, 0

    def forward_pass(self, raw):
        self.reset_values()
        self.y1, self.x1 = raw[0], raw[0]
        self.y0, self.x0 = raw[1], raw[1]

        filtered = [self.y1, self.y0]
        for i in range(2, len(raw)):
            f = self.filter_sample(raw[i])
            filtered.append(f)
        return filtered

    def backward_pass(self, forward):
        self.reset_values()
        self.y1, self.x1 = forward[-1], forward[-1]
        self.y0, self.x0 = forward[-2], forward[-2]

        filtered = [self.y1, self.y0]
        for i in range(len(forward) - 3, -1, -1):
            f = self.filter_sample(forward[i])
            filtered.append(f)
        return filtered[::-1]

    def filter_samples_internal(self, samples, fs, fc):
        self.update_coefficients(fs, fc)
        forward = self.forward_pass(samples)
        backward = self.backward_pass(forward)
        return backward

    def filter_sample(self, sample):
        self.x2, self.x1, self.x0 = self.x1, self.x0, sample
        self.y2, self.y1 = self.y1, self.y0
        self.y0 = (
            self.a0 * self.x0
            + self.a1 * self.x1
            + self.a2 * self.x2
            + self.b1 * self.y1
            + self.b2 * self.y2
        )
        return self.y0

    def add_padding(self, samples, padding):
        padded = np.zeros(len(samples) + 2 * padding)
        for i in range(padding):
            padded[i] = samples[0] + samples[0] - samples[padding - i]
        for i in range(len(samples)):
            padded[padding + i] = samples[i]
        for i in range(padding):
            padded[padding + len(samples) + i] = (
                samples[-1] + samples[-1] - samples[len(samples) - 2 - i]
            )
        return padded

    def remove_padding(self, samples, padding):
        return samples[padding:-padding]

    def durbin_watson(self, residuals):
        diff = np.diff(residuals)
        numerator = sum(diff**2)
        denominator = sum(residuals**2)
        dw = numerator / denominator
        return dw


class FilteringResult:
    def __init__(self, fc, data, dw_normalized):
        self.fc = fc
        self.data = data
        self.dw_normalized = dw_normalized


class FilteredTrajectory:
    def __init__(self):
        self.length = 0
        self.can_filter = False
        self.times = []
        self.raw_xs = []
        self.raw_ys = []
        self.filter_result_xs = []
        self.x_cutoff_index = -1
        self.filter_result_ys = []
        self.y_cutoff_index = -1
        self.best_fit_circle = None

    def initialize(self, x, y, calibration_helper=None):
        self.length = len(x)
        self.raw_xs = x
        self.raw_ys = y

        self.can_filter = len(x) > 10

        if self.can_filter:
            # Framerate is set to 30 for this example, replace with your actual framerate
            framerate = 30

            filter = ButterworthFilter()

            tests = 10
            best_cutoff_index_x = 0
            self.filter_result_xs, best_cutoff_index_x = filter.filter_samples(
                self.raw_xs, framerate, tests
            )
            self.x_cutoff_index = best_cutoff_index_x

            best_cutoff_index_y = 0
            self.filter_result_ys, best_cutoff_index_y = filter.filter_samples(
                self.raw_ys, framerate, tests
            )
            self.y_cutoff_index = best_cutoff_index_y

        # I ADDED THIS 3 LINES TO GET THE FILTERED DATA
        # Check if the best cutoff indices are within valid range before accessing data
        if 0 <= self.x_cutoff_index < len(self.filter_result_xs):
            x_data = self.filter_result_xs[self.x_cutoff_index].data
        else:
            x_data = self.raw_xs

        if 0 <= self.y_cutoff_index < len(self.filter_result_ys):
            y_data = self.filter_result_ys[self.y_cutoff_index].data
        else:
            y_data = self.raw_ys
        return x_data, y_data

        # TODO: Calculate the best fit circle here if needed


if __name__ == "__main__":
    # Create an instance of FilteredTrajectory
    trajectory = FilteredTrajectory()

    # Initialize the trajectory with your data, e.g., a list of x and y values
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    trajectory.initialize(x, y)

    x_data = trajectory.filter_result_xs[trajectory.x_cutoff_index].data
    y_data = trajectory.filter_result_ys[trajectory.y_cutoff_index].data
