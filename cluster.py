import face_recognition
import cv2
import glob
import os
import numpy as np
import dlib

class cluster() :

	def __init__(self,img_path) :

		self.img_path = img_path
		self.faces_info = []
		self.face_encodings = []
		self.img_files = []
		
		for ext in ('*.png', '*.jpg', '*.jpeg') :
			self.img_files.extend(glob.glob(self.img_path+'/'+ext))

	def encodings(self) :
		
		print("[info]  Encodings faces.......")

		for self.img in self.img_files :
			img_name = self.img.split('/')[-1].split('.')[0]
			extension = self.img.split('/')[-1].split('.')[1]
			print("Processing image : {}".format(img_name+'.'+extension))

			img_load = face_recognition.load_image_file(self.img)
			face_locs = face_recognition.face_locations(img_load, number_of_times_to_upsample=1, model='hog')
			img_encodes = face_recognition.face_encodings(img_load,known_face_locations=face_locs, num_jitters=1)
			
			for (face_loc, encoding) in zip(face_locs, img_encodes) :
				info = [{'img_path' : self.img, 'face_location' : face_loc, 'face_encoding' : encoding}]
				self.faces_info.extend(info)
			

		self.clustring(self.faces_info)

	def clustring(self, faces_info) :

		for data in faces_info :
			encode = data['face_encoding']
			self.face_encodings.append(dlib.vector(encode))

		labels = dlib.chinese_whispers_clustering(self.face_encodings, 0.5)
		labels = np.array(labels)
		print("All cluster labels :", labels)
		
		unique_labels = np.unique(labels)
		print("Number of unique faces found  : ", len(unique_labels))
		print("Saving faces..........")
		for label in unique_labels :
			index = np.where(labels == label)[0]

			for i in index :
				image_path = self.faces_info[i]['img_path']
				image_name = image_path.split('/')[-1].split('.')[0]
				image_ext = image_path.split('/')[-1].split('.')[1]
				image = cv2.imread(image_path)

				output_dir = os.getcwd()+'/'+str(label)

				if not os.path.isdir(output_dir) :
					os.mkdir(str(label))
		
				cv2.imwrite(output_dir+'/'+image_name+'.'+image_ext, image)

if __name__ == '__main__' :

	path = input("Enter the path of image directory : ")
	image = cluster(path)
	image.encodings()