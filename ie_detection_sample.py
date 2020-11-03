from __future__ import print_function
import argparse
import os
import cv2
import numpy as np
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork, InferRequest

class InferenceEngineDetector: 
	def __init__(self, configPath = None, weightsPath = None, extension=None, classesPath = None): 
		self.ie_core = IECore()
		self.net = self.ie_core.read_network(model=configPath, weights=weightsPath)
		self.exec_net= self.ie_core.load_network(network=self.net, device_name='CPU')

	def _prepare_image(self, image, h, w):
		#image = cv2.imread(imagePath)
		image = cv2.resize(image,(w, h))
		image = image.transpose((2, 0, 1))
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#blob = np.expand_dims(image, axis=0)
		#return blob
		return image

	def detect(self, imagePath): 
		input_blob = next(iter(self.net.inputs)) 
		out_blob = next(iter(self.net.outputs))
		n, c, h, w = self.net.inputs[input_blob].shape

		blob = self._prepare_image(imagePath, h, w)

		output = self.exec_net.infer(inputs = {input_blob: blob})
		output = output[out_blob]

		return output

	def draw_detection(self, detections, image, confidence=0.1, draw_text=True):
		data = detections[0][0]
		w, h = image.shape[:-1]
		
		for number, proposal in enumerate(data):
			if proposal[2] > confidence:
				label = np.int(proposal[1])
				xmin = np.int(w * proposal[3])
				ymin = np.int(h * proposal[4])
				xmax = np.int(w * proposal[5])
				ymax = np.int(h * proposal[6]) 
				

				point1 = (xmin, ymax)
				point2 = (xmax, ymin)
				color = (232, 35, 244)

				image_detected = cv2.rectangle(image, point1, point2, color, 2)

				if draw_text:
					text = f"Class: {label}"
					cv2.putText(image, text, (0,11), cv2.FONT_HERSHEY_COMPLEX, 0.45, color, 1)
				
				return image_detected           


        
        
        
        
def build_argparser():
	parser = argparse.ArgumentParser() 
	parser.add_argument('-m', '--model', help = 'Path to an .xml \ file with a trained model.', required = True, type = str) 
	parser.add_argument('-w', '--weights', help = 'Path to an .bin file \ with a trained weights.', required = True, type = str) 
	parser.add_argument('-i', '--input', help = 'Path to \ image file', required = True, type = str) 
	parser.add_argument('-l', '--cpu_extension', help='MKLDNN \ (CPU)-targeted custom layers. Absolute path to a shared library \ with the kernels implementation', type=str, default=None) 
	parser.add_argument('-c', '--classes', help = 'File containing \ classnames', type = str, default = None) 

	return parser

def main(): 
	args = build_argparser().parse_args() 
	ie_detector = InferenceEngineDetector(configPath=args.model, weightsPath=args.weights, extension=args.cpu_extension, classesPath=args.classes) 
	img = cv2.imread(args.input)

	detections = ie_detector.detect(img) 
    
	image_detected = ie_detector.draw_detection(detections, img) 

	cv2.imshow('Image with detections', image_detected) 
	cv2.waitKey(0) 
	cv2.destroyAllWindows() 
    
	return

if __name__ == '__main__': 
		sys.exit(main())