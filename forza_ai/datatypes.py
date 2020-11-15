from imageai.Detection import ObjectDetection
from PIL import Image
from typing import Tuple
import numpy as np
import pygame


class RecognizedObject:

    def __init__(self, name: str, confidence: float, points: list):
        self.name = name
        self.confidence = confidence
        self.points = [int(p / 2) for p in points]
        # change x2, y2 to width, height
        self.points[2] -= self.points[0]
        self.points[3] -= self.points[1]
        self.rect = pygame.Rect(*self.points)  # left, top, width, height
        self.is_primary = self.rect.collidepoint(475, 410)

    def __repr__(self):
        return repr(f'<RecognizedObject{"[primary]" if self.is_primary else ""} name={self.name} points={self.points}>')

    @staticmethod
    def from_dict(dic: dict):
        return RecognizedObject(dic['name'], dic['percentage_probability'], dic['box_points'])


class PredictionData:

    def __init__(self, original: Image, segmentation: np.ndarray, lines: np.ndarray = None, assoc_info: dict = None):
        """
        A helper class used to store data from the prediction model.

        :param original: The original screen capture.
        :param segmentation: Segmentation data.
        :param lines: An array of detected road markings.
        :param assoc_info: Other data to be associated with this prediction.
        """
        self.original = original
        self.segmentation = segmentation
        self.lines = lines
        self.assoc_info = assoc_info or {}

        self.size: Tuple[int, int] = original.size

    def _minmax(self, num: float, min_n: int, max_n: int) -> int:
        return int(max(min(num, max_n), min_n))

    def _constrain_pos(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope, y_intercept = params[0], params[1]

        if slope < 0:
            x1 = self._minmax(x1, 0, int(self.size[0] / 2))
            x2 = self._minmax(x2, 0, int(self.size[0] / 2))
        else:
            x1 = self._minmax(x1, int(self.size[0] / 2), self.size[0])
            x2 = self._minmax(x2, int(self.size[0] / 2), self.size[0])

        y1 = (slope * x1) + y_intercept
        y2 = (slope * x2) + y_intercept

        return x1, y1, x2, y2

    @property
    def x_center(self) -> int:
        """
        Get the center of the screen on the x axis.
        """
        return int(self.size[0] / 2)

    @property
    def x_offset(self) -> float:
        """
        Compute the offset of the two edge lines from the center of the screen.
        """
        x2_l = self._minmax(self.lines[0][2], 0, self.size[0])
        x2_r = self._minmax(self.lines[1][2], 0, self.size[0])
        return ((x2_l + x2_r) / 2) - self.x_center

    @staticmethod
    def _as_pygame_image(image: Image, size: Tuple[int, int] = None) -> pygame.Surface:
        """
        Internal method to convert a PIL image to a PyGame surface.

        :param image: The PIL image to be converted.
        :param size: Tuple of (width, height) to change the image size to, if any.
        :return: A PyGame `Surface`.
        """
        if size:
            image.thumbnail(size, Image.ANTIALIAS)
        to_blit_bytes = image.tobytes()

        return pygame.image.fromstring(to_blit_bytes, size, image.mode)

    def original_as_pygame_image(self, size: Tuple[int, int] = None) -> pygame.Surface:
        """
        Convert the original screen capture to a PyGame surface.
        """
        return self._as_pygame_image(self.original, size)

    def segmentation_as_pil_image(self) -> Image:
        """
        Convert segmentation data to a PIL image.
        """
        return Image.fromarray(self.segmentation)

    def segmentation_as_pygame_image(self, size: Tuple[int, int] = None) -> pygame.Surface:
        """
        Convert segmentation data to a PyGame surface.
        """
        seg_as_pil = self.segmentation_as_pil_image().convert('RGB')
        return self._as_pygame_image(seg_as_pil, size)

    def visualize_lines(self) -> pygame.Surface:
        """
        Visualize lane detection data.
        """
        surf = pygame.Surface(self.size, pygame.SRCALPHA, 32)
        if self.lines is not None:
            for x1, y1, x2, y2 in self.lines:
                print(f'b: {x1} {y1}, {x2} {y2}')
                x1, y1, x2, y2 = self._constrain_pos(x1, y1, x2, y2)
                print(f'a: {x1} {y1}, {x2} {y2}')

                # draw line along prediction average
                pygame.draw.line(surf, (0, 255, 0), (x1, y1), (x2, y2), 5)
                # line from first coordinate to prediction apex
                pygame.draw.line(surf, (255, 0, 0), (x1, y1), (self.x_center, 450), 5)

        # steering axis lines
        pygame.draw.line(surf, (0, 0, 255), (self.x_center, self.size[1]), (self.x_center, 0), 5)
        pygame.draw.line(surf, (255, 0, 255), (self.x_center, self.size[1]), (int(self.x_offset + self.x_center), 450), 5)

        return surf

    def do_object_recognition(self, detector: ObjectDetection, objects: dict) -> Tuple[pygame.Surface, list]:
        out, recog = detector.detectCustomObjectsFromImage(
            custom_objects=objects,
            input_image=np.array(self.original),
            input_type='array',
            output_type='array')
        out_img = Image.fromarray(np.uint8(out)).convert('RGB')
        pg_img = self._as_pygame_image(out_img, (960, 540))

        recog = [RecognizedObject.from_dict(o) for o in recog]

        return pg_img, recog

