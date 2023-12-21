from .calibrationPlane import CalibratorLine, CalibratorPlane


class CalibrationHelper:
    def __init__(self, image_size, calibrator_type='Line'):
        self.image_size = image_size
        self.calibrator_type = calibrator_type
        self.calibrator = None  # Initialize calibrator based on the calibrator_type

    def initialize_calibrator(self, quad):
        if self.calibrator_type == 'Line':
            self.calibrator = CalibratorLine(quad)
        elif self.calibrator_type == 'Plane':
            self.calibrator = CalibratorPlane(quad)

    def set_calibration_origin(self, p):
        if self.calibrator:
            self.calibrator.set_origin(p)

    def transform_point(self, point):
        if self.calibrator:
            return self.calibrator.transform(point)
        else:
            return point

    def get_length_text(self, p1, p2, precise=False, abbreviation=False):
        length = self.get_distance(self.transform_point(p1), self.transform_point(p2))
        return self.format_value(length, precise) + self.length_abbreviation(abbreviation)

    def format_value(self, value, precise=False):
        if precise:
            return '{:.2f}'.format(value)
        else:
            return '{:.0f}'.format(value)

    def length_abbreviation(self, abbreviation=False):
        if abbreviation:
            return " " + self.length_unit_abbreviation()
        else:
            return ""

    def length_unit_abbreviation(self):
        return "meter"
