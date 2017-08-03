from flask import Flask, request, render_template
import training
import os
import tensorflow as tf
import time

LOG_DIR = './train/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
ckpt = tf.train.get_checkpoint_state(LOG_DIR)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            max_index, prediction = training.evaluate_one_image(ckpt, file)
            if max_index == 0:
                result = '猫'
            else:
                result = '狗'

            filename = str(time.time()) + '.' + file.filename.rsplit('.', 1)[1]
            file.save(os.path.join('./uploads/', filename))
            return render_template('result.html', name=result)


    else:
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
