from imutils import paths
import face_recognition
import pickle
import cv2
import os
from google.colab.patches import cv2_imshow

IMAGE_RES = (256, 256)
font = cv2.FONT_HERSHEY_COMPLEX # Text in video
font_size = 0.4
blue = (225,0,0)
green = (0,128,0)
red = (0,0,255)
orange = (0,140,255)

class FaceRecognizer:
  def __init__(self, faceCascadePath, encodingsPath) -> None:
    self._faceCascade = cv2.CascadeClassifier(faceCascadePath)
    self._encodingsPath = encodingsPath
    self._knownEncodings = []
    self._knownNames = []
    self._data = {}
 
  def learn_faces_batch(self, imagePaths):
    for imagePath in imagePaths:
      name = imagePath.split(os.path.sep)[-2]
      self.learn_face(name, imagePath)
    self._data = pickle.loads(open(self._encodingsPath, "rb").read())

  def learn_face(self, name, imagePath):
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    image = cv2.resize(image, IMAGE_RES)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Use Face_recognition to locate faces
    face_locations = face_recognition.face_locations(rgb, model='cnn')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, face_locations)

    # loop over the encodings
    for encoding in encodings:
      self._knownEncodings.append(encoding)
      self._knownNames.append(name)
    self._save_encodings()

  def _save_encodings(self):
    # save encodings along with their names in dictionary data
    data = {"encodings": self._knownEncodings, "names": self._knownNames}
    # use pickle to save data into a file for later use
    f = open(self._encodingsPath, "wb")
    f.write(pickle.dumps(data))
    f.close()

  def annotated_frame(self):
    return self._output_frame

  def reco_list(self):
    return self._names

  def refresh(self, frame, faces, frame_to_draw):
    self._frame = frame.copy()
    self._faces = faces.copy()
    self._draw_frame = frame_to_draw
    self._analyze()

  def _analyze(self):
    frame = self._frame
    frame = cv2.flip(frame, 1)

    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # the facial encodings for face in input
    face_locations = face_recognition.face_locations(rgb, model='cnn')
    encodings = face_recognition.face_encodings(rgb, face_locations)
    names = []

    # loop over the facial encodings incase
    # we have multiple encodings for multiple fcaes
    for encoding in encodings:
      # Compare encodings with encodings in data["encodings"]
      # Matches contain array with boolean values and True for the encodings it matches closely
      # and False for rest
      matches = face_recognition.compare_faces(self._data["encodings"],
        encoding)
      # set name = unknown if no encoding matches
      name = "Unknown"
      # check to see if we have found a match
      if True in matches:
          # Find positions at which we get True and store them
          matchedIdxs = [i for (i, b) in enumerate(matches) if b]
          counts = {}
          # loop over the matched indexes and maintain a count for
          # each recognized face face
          for i in matchedIdxs:
              # Check the names at respective indexes we stored in matchedIdxs
              name = self._data["names"][i]
              # increase count for the name we got
              counts[name] = counts.get(name, 0) + 1
          # set name which has highest count
          name = max(counts, key=counts.get)

      # update the list of names
      names.append(name)
      # loop over the recognized faces
      for ((x, y, w, h), name) in zip(self._faces, names):
          # rescale the face coordinates
          # draw the predicted face name on the image
          cv2.rectangle(self._draw_frame, (int(x), int(y)), (int(w), int(h)), orange, 2)
          cv2.putText(self._draw_frame, name, (int(x) + 2, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
          
    self._names = names
    self._output_frame = self._draw_frame
