import argparse
import io
from flask import Flask, request, jsonify
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model
from utils import prepare, decode, heat_map
import tensorflow as tf

DETECTION_URL = "/"
app = Flask(__name__)

labels = ['Normal', 'Not Normal']
sublabels = ['Cardiomegaly', 'Hernia', 'Infiltration', 'Nodule', 'Emphysema',
             'Effusion', 'Atelectasis', 'Pleural_Thickening', 'Pneumothorax', 'Mass',
             'Fibrosis', 'Consolidation', 'Edema', 'Pneumonia']
input_shape = (224, 224, 3)
img_input = Input(shape=input_shape)
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=img_input, input_shape=input_shape)          #базовая модель
x = base_model.output
x = Flatten(input_shape=base_model.output_shape[1:])(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid", name="predictions")(x)


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        base_img = prepare(img)
        prediction = model.predict([base_img])                                                  #дигноз базовой модели: normal/not normal
        results = jsonify(str(labels[round(prediction[0][0])]))
        if args.ensemble_models and round(prediction[0][0]):
            sub_img = decode(image_bytes)
            print(img)
            pred = submodel.predict(sub_img)
            idx = (-pred).argsort()[0][:3]                                                      # берёт индексы 3 самых вероятных заболеваний
            print(idx)
            disease = ''
            heat_map_img = heat_map(image_bytes, last_conv_layer, submodel,args.heat_intencity)
            for i in idx:
                print(i)
                disease += sublabels[i] + ' prob=' + str(pred[0][i])[:4] + ', '
            return jsonify(disease[:-2])
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--heat_intencity", default=0.2, type=float, help='heat map intencity [0:1]')
    parser.add_argument("--ensemble_models", default=1, type=int, help="activate disease classification")
    args = parser.parse_args()


    model = Model(inputs=img_input, outputs=predictions)
    model.load_weights('vgg16_rgb_binary_xray_class_equal_weights.best.hdf5')

    if args.ensemble_models == True:
        submodel = tf.keras.models.load_model('NIH_Seresnet152_model.h5')                       # иницивализация модели для диагнозов
        last_conv_layer = submodel.get_layer('dropout')                                         # последний свёрточный слой - через model.summary(), нужен для heatmap

    app.run(host="0.0.0.0", port=args.port)

