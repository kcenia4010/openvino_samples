from __future__ import print_function
import argparse
import os
import cv2
import numpy as np
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, ExecutableNetwork, InferRequest


class InferenceEngineClassifier: 
	def __init__(self, configPath = None, weightsPath = None, classesPath = None): 
		self.ie_core = IECore()
		self.net = self.ie_core.read_network(model=configPath, weights=weightsPath)
		self.exec_net= self.ie_core.load_network(network=self.net, device_name='CPU')

	def get_top(self, prob, topN = 1):
		prob = np.squeeze(prob)
		prob2 = np.argsort(prob)
		predictions = []
		for i in range(topN):
		    predictions.insert(0, {prob2[i-topN], prob[prob2[i-topN]]})
		return predictions


	def _prepare_image(self, image, h, w):
		#image = cv2.imread(imagePath)
		image = cv2.resize(image,(w, h))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.transpose((2, 0, 1))
		blob = np.expand_dims(image, axis=0)
		return blob

	def classify(self, imagePath): 
		input_blob = next(iter(self.net.input_info)) 
		out_blob = next(iter(self.net.outputs))
		n, c, h, w = self.net.inputs[input_blob].shape

		blob = self._prepare_image(imagePath, h, w)

		output = self.exec_net.infer(inputs = {input_blob: blob})
		output = output[out_blob]
		return output
        
        
def build_argparser(): 
	parser = argparse.ArgumentParser() 
	parser.add_argument('-m', '--model', help = 'Path to an .xml \ file with a trained model.', required = True, type = str) 
	parser.add_argument('-w', '--weights', help = 'Path to an .bin file \ with a trained weights.', required = True, type = str) 
	parser.add_argument('-i', '--input', help = 'Path to \ image file', required = True, type = str) 

	return parser

def main(): 
	log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout) 
	args = build_argparser().parse_args() 
	log.info("Start IE classification sample") 
	ie_classifier = InferenceEngineClassifier(configPath=args.model, weightsPath=args.weights) 
	img = cv2.imread(args.input) 
	prob = ie_classifier.classify(img) 
	predictions = ie_classifier.get_top(prob, 5) 
	log.info("Predictions: " + str(predictions)) 
	return 

if __name__ == '__main__': 
		sys.exit(main())