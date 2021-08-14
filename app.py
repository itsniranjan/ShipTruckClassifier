from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)

dic = {0: 'Ship', 1: 'Truck'}

model = load_model('shiptruck.h5')

model.make_predict_function()


def predict_label(img_path, model):
    i = image.load_img(img_path, target_size=(32, 32))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 32, 32, 3)
    p = model.predict(i)
    return p


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Hi guys! This is all about me"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(basepath, 'images',
                                secure_filename(img.filename))
        img.save(img_path)

        p = predict_label(img_path, model)
        if(p[0][0] < 0.5):
            text = "ITS A TRUCK"
        else:
            text = "ITS A SHIP"

    return render_template("index.html", prediction=text, img_path=img_path.replace(os.sep, '/'))


if __name__ == '__main__':
    app.run(debug=True,port=int(os.environ.get('PORT', 3000)))
