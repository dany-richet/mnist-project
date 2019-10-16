from PIL import Image
import joblib

class Model:

    def __init__(self, path_pickle):
        self.path_pickle = path_pickle
        self.sklearn_model = self.load_model()

    def predict(self, img_path):
        
        try:
            img  = joblib.load(img_path)
        except IOError:
            raise
        
        prediction = self.sklearn_model.predict(img)
        return prediction

    def load_model(self):

        pickle_model = joblib.load(self.path_pickle)
        return pickle_model
