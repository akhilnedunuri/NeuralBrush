from flask import Flask, render_template, request
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)


model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")


UPLOAD_FOLDER = "static/images/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_image(img_path):
    
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (400, 400))  
    img = tf.expand_dims(img, axis=0)  
    return img


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        content_file = request.files.get("content")
        style_file = request.files.get("style")

        if content_file and style_file:
            
            content_path = os.path.join(app.config["UPLOAD_FOLDER"], "content.jpg")
            style_path = os.path.join(app.config["UPLOAD_FOLDER"], "style.jpg")
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], "stylized.jpg")

            content_file.save(content_path)
            style_file.save(style_path)

            
            content_image = load_image(content_path)
            style_image = load_image(style_path)

            
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

            # Convert tensor to image
            output_image = np.array(stylized_image[0] * 255, dtype=np.uint8) 
            output_pil = Image.fromarray(output_image)
            output_pil.thumbnail((500, 500))  # Resize for better display
            output_pil.save(output_path)

            return render_template(
                "index.html",
                content_img="content.jpg",
                style_img="style.jpg",
                output_img="stylized.jpg",
            )

    return render_template("index.html", content_img=None, style_img=None, output_img=None)


if __name__ == "__main__":
    app.run(debug=True)
