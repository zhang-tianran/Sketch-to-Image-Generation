import cv2
from scipy.spatial.distance import cosine
import torchvision.models as models
import torch
import numpy as np

class evaluation():
	def __init__(self):
		self.model = models.vgg16(pretrained=True)
		classifier = list(self.model.classifier)[:-1]
		classifier = torch.nn.Sequential(*classifier)
		self.model.classifier = classifier
		self.model.eval()
	
	def get_feature_vector(self, img):
		img = cv2.resize(img, (224, 224))
		img = img.reshape(1, 3, 224, 224)
		img = torch.Tensor(img)
		feature_vector = self.model(img)
		return feature_vector.detach().numpy()
	
	def calculate_similarity(self, img1, img2):
		vector1 = self.get_feature_vector(img1)
		vector2 = self.get_feature_vector(img2)
		return 1 - cosine(vector1, vector2)

	def similarity(self, pred, labels):
		sim = 0
		for i in range(pred.shape[0]):
			sim += self.calculate_similarity(pred[i], labels[i])
		return sim / pred.shape[0]

	def test(self, model, test_input, test_labels):
		sim = 0
		i = 0
		while i + model.batch_size <= test_input.shape[0]:
			batch_input = test_input[i: i+model.batch_size]
			batch_label = test_labels[i: i+model.batch_size]
			pred = model.call(batch_input)
			sim += self.similarity(pred, batch_label)
			i += model.batch_size
		return sim * model.batch_size / i

if __name__ == '__main__':
	model = evaluation()

	img1 = cv2.imread("./img1.jpg")
	img2 = cv2.imread("./img2.jpg")
	img3 = cv2.imread("./img3.jpg")
	img4 = cv2.imread("./img4.jpg")
	imgs = np.array([img1, img2, img3, img4])
	labels = np.array([img1, img1, img1, img1])
	
	print(model.calculate_similarity(img1, img2))
	print(model.calculate_similarity(img1, img3))
	print(model.calculate_similarity(img1, img4))


# import cv2
# from scipy.spatial.distance import cosine
# from keras.models import Model
# import tensorflow as tf
# import numpy as np

# class evaluation():
# 	def __init__(self):
# 		vgg16 = tf.keras.applications.VGG16(weights='imagenet')
# 		self.model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
		
# 	def get_feature_vector(self, img):
# 		img = cv2.resize(img, (224, 224))
# 		feature_vector = self.model(img.reshape(1, 224, 224, 3))
# 		return feature_vector
	
# 	def calculate_similarity(self, img1, img2):
# 		vector1 = self.get_feature_vector(img1)
# 		vector2 = self.get_feature_vector(img2)
# 		return 1 - cosine(vector1, vector2)

# 	def similarity(self, pred, labels):
# 		sim = 0
# 		for i in range(pred.shape[0]):
# 			sim += self.calculate_similarity(pred[i], labels[i])
# 		return sim / pred.shape[0]

# 	def test(self, model, test_input, test_labels):
# 		sim = 0
# 		loss = 0
# 		i = 0
# 		while i + model.batch_size <= test_input.shape[0]:
# 			batch_input = test_input[i: i+model.batch_size]
# 			batch_label = test_labels[i: i+model.batch_size]
# 			pred = model.call(batch_input)
# 			loss += model.loss_function(pred, batch_label)
# 			sim += self.similarity(pred, batch_label)
# 			i += model.batch_size
# 		return np.exp(loss * model.batch_size / i), sim * model.batch_size / i

# if __name__ == '__main__':
# 	model = evaluation()

# 	img1 = cv2.imread("./img1.jpg")
# 	img2 = cv2.imread("./img2.jpg")
# 	img3 = cv2.imread("./img3.jpg")
# 	img4 = cv2.imread("./img4.jpg")
# 	imgs = np.array([img1, img2, img3, img4])
# 	labels = np.array([img1, img1, img1, img1])
	
# 	print(model.calculate_similarity(img1, img2))
# 	print(model.calculate_similarity(img1, img3))
# 	print(model.calculate_similarity(img1, img4))