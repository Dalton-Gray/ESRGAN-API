import unittest
import torch
import numpy
import cv2
from image_utilities import prepare_image, image_to_numpy, output_to_image, save_image_to_output_folder
from models import upscale_image, RRDB_PSNR, interp08
import torch_testing as tt
from PIL import Image
import shutil



class TestValues(object):
    def __init__(self):
        shutil.copy('static/input/0801_Validation_Area_downscaled.png', 'testing')
        
        image_location = 'testing/0801_Validation_Area_downscaled.png'

        self.device = torch.device('cpu')
        self.test_output_folder = 'testing'
        self.test_image = Image.open(image_location)
        self.test_image_numpy = cv2.imread(image_location) * 1.0 / 255 
        self.test_tensor_image = torch.from_numpy(numpy.transpose(self.test_image_numpy[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(self.device)
        self.test_numpy_array = numpy.ones([100,100,3],dtype=numpy.uint8)
        self.test_transposed_numpy_array = numpy.ones([100,3,3],dtype=numpy.uint8)

        self.test_RRDB_PSNR = RRDB_PSNR(self.device)
        self.test_interp08 = interp08(self.device)
        

class PositiveTests(unittest.TestCase):

    def setUp(self):
        super(PositiveTests, self).setUp()
        self.values = TestValues()

    def test_prepare_image(self):
        tt.assert_equal(prepare_image(self.values.test_image_numpy, self.values.device), self.values.test_tensor_image)

    def test_image_to_numpy(self):
        self.assertEqual(image_to_numpy(self.values.test_image).tolist(), self.values.test_image_numpy.tolist())
    
    def test_output_to_image(self):
        self.assertEqual(output_to_image(self.values.test_numpy_array).all(), self.values.test_transposed_numpy_array.all())

    def test_save_image_to_output_folder(self):
        self.assertEqual(save_image_to_output_folder(self.values.test_image, self.values.test_image_numpy, self.values.test_output_folder),  "Image saved to folder")

class NegativeTests(unittest.TestCase):

    def setUp(self):
        super(NegativeTests, self).setUp()
        self.values = TestValues()

    def test_prepare_image(self):
        self.assertEqual(prepare_image([], self.values.device), "TypeError when preparing image for upscaling")

    # def test_prepare_image_missing_value(self):
    #     self.assertEqual(prepare_image(None, self.values.device), "Exception when preparing image for upscaling")

    def test_image_to_numpy_attribute_error(self):
        self.assertEqual(image_to_numpy([]), "AttributeError when converting image to numpy")
    
    def test_output_to_image(self):
        self.assertEqual(output_to_image([]), "TypeError when converting model output to image")

    def test_save_image_to_output_folder(self):
        self.assertEqual(save_image_to_output_folder([], 0, self.values.test_output_folder),  "AttributeError when saving image to folder")


if __name__ == '__main__':
    unittest.main()
