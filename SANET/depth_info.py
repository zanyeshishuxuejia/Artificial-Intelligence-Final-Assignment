import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# Load model
def load_model(model_path):
    print('Loading model...')
    model = torch.load(model_path)
    return model


def my_DepthNorm(x, maxDepth):
    return maxDepth / x


def my_predict(model, images, minDepth=10, maxDepth=1000):

    with torch.no_grad():
        # Compute predictions
        predictions = model(images)

        # Put in expected range
    return np.clip(my_DepthNorm(predictions.numpy(), maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


# Input images
def input_images(image):
    pytorch_input = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    print(pytorch_input.shape)
    print('\nLoaded (1) image of size{0}.'.format(image.shape))
    return pytorch_input


# Compute results
def compute_results(pytorch_input, model):
    output = my_predict(model, pytorch_input[0, :, :, :].unsqueeze(0))
    return output[0, 0, :, :]


if __name__ == '__main__':

    pytorch_model = load_model('torch_model.pkl')
    pytorch_model.eval()
    x = np.clip(np.asarray(Image.open('./examples/11_image.png'),
                dtype=float) / 255, 0, 1).astype('float32')
    torch_inp = input_images(x)
    depth_img = compute_results(torch_inp, pytorch_model)
    plt.imshow(depth_img)
    plt.show()
