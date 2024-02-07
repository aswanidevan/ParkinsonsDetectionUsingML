from flask import Flask,render_template,request
import cv2
import os
from skimage import feature
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from werkzeug.utils import secure_filename
from xgboost import XGBClassifier

UPLOAD_FOLDER = './upload'

app=Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/create_user')
def create_user():
    return render_template('create-user.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    pred="Choose your handrawn spiral/wave sample image and click Predict to check the results."
    if request.method == "POST":
        select = request.form.get("image-type")
        file = request.files['uploadedfile']
        filename = secure_filename(file.filename)
        basepath=os.path.dirname(__file__)
        testingPath=os.path.join(basepath, "uploads",filename)
        if file :
            file.save(testingPath)
            image = cv2.imread(testingPath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (200, 200))
            image = cv2.threshold(image, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
            features = feature.hog(image, orientations=9,
                               pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                               transform_sqrt=True, block_norm="L1")
                
            if(str(select)=="wave"):
                model = joblib.load("waveparkinson.pkl")
            elif(str(select)=="spiral"):
                model = joblib.load("spiralparkinson.pkl")
            else:
                return render_template('predict.html',prediction=pred)
    
            #model = pickle.loads(open("waveparkinson.pkl","wb").read())
            preds=model['Rf']['classifier'].predict([features])     
            pred="You have been diagnosed with Parkinson's Disease" if preds else "No worries you are Healthy"      
    
        

    return render_template('predict.html',prediction=pred)

@app.route('/reports')
def reports():
    return render_template('reports.html')



if __name__ == '__main__':
    app.run(debug=True)
