import numpy as np
import pickle
from PIL import Image

results = pickle.load(open("results.pkl",'rb'))
res_array = np.asarray(results[0])
print(np.max(res_array))
height = np.shape(res_array)[0]
width = np.shape(res_array)[1]

data = np.zeros((height, width,3), dtype=np.int8)
for i in range(3):
    data[:, :, i] = res_array

shapeddata = data.reshape((width*height,3))
np.power(shapeddata, 1/2.2)
np.multiply(np.divide(shapeddata, np.max(res_array)),255)
output = Image.new("RGB", (width, height))
output.putdata([tuple(pixel) for pixel in shapeddata])
rotated = output.transpose(Image.ROTATE_90)
rotated.save("output.png")
