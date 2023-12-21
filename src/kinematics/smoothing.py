import numpy as np


class MovingAverage:
    def filter_samples(self, samples, fs, span, sentinels=1):
        interval = 1000 / fs

        if span / 2 <= interval:
            return samples

        padding = 50
        padding = max(0, min(padding, len(samples) - (sentinels * 4)))
        padded = self.add_padding(samples, padding, sentinels)

        frames = int(span / (2 * interval))

        smoothed = np.convolve(padded, np.ones(frames) / frames, mode='same')
        return self.remove_padding(smoothed, padding)

    def add_padding(self, samples, padding, sentinels):
        pivot = sentinels
        left_padding = samples[pivot: pivot + padding]
        right_padding = samples[-sentinels - padding: -sentinels]

        reflected_left = samples[pivot] + (samples[pivot] - left_padding)
        reflected_right = samples[-sentinels - 1] + \
            (samples[-sentinels - 1] - right_padding)

        padded = np.concatenate((reflected_left, samples, reflected_right))
        return padded

    def remove_padding(self, samples, padding):
        return samples[padding:len(samples) - padding]
