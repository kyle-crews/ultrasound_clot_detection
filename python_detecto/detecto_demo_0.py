# import detecto's modules
from detecto import core, utils, visualize

# read the image and initialize a pre-trained model
image = utils.read_image('./ultrasound_clot_detection/python_detecto/images/zebra.jpg')
model = core.Model()

# generate and plot the top perditions
labels, boxes, scores = model.predict_top(image)
visualize.show_labeled_image(image, boxes, labels)