import sys

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


def main():
    if len(sys.argv) < 2:
        print(
            'ERROR: Image filename not provided.\n'
            'USAGE: python main.py [filename]', file=sys.stderr)
    else:

        try:
            img = Image.open(sys.argv[1])
            img = ImageOps.grayscale(img)
            img = (tf.convert_to_tensor(img, dtype=tf.float32) / 255.0).numpy()
            img = img[None, ...]

            model = tf.keras.models.load_model('my_model')
            print(f'PREDICTION: {np.argmax(model(img))}')

        except FileNotFoundError:
            print(f'ERROR: Cannot find file "{sys.argv[1]}"\n'
                  'USAGE: python main.py [filename]',
                  file=sys.stderr)
        except IOError:
            print(f'ERROR: Model not trained!\n'
                  'USAGE: To train the model, run "python train.py"',
                  file=sys.stderr)
        except:
            print('ERROR: An unknown error occurred.',
                  file=sys.stderr)


if __name__ == '__main__':
    main()
