import numpy as np
import pickle
from PIL import Image

results = pickle.load(open("results.pkl",'rb'))
res_array = np.asarray(results[0])
arraymax = np.max(res_array)
height = np.shape(res_array)[0]
width = np.shape(res_array)[1]

data = np.zeros((height, width,3), dtype=np.int8)
for i in range(3):
    data[:, :, i] = res_array

normalized = np.power(data, 1/2.2)
normmax = np.power(arraymax, 1/2.2)
shapeddata = normalized.reshape((width*height,3))
scaled = np.multiply(np.divide(shapeddata, normmax),255)
result = np.rint(scaled)
print(result)
output = Image.new("RGB", (width, height))
output.putdata([tuple([int(pixel[0]),int(pixel[1]),int(pixel[2])]) for pixel in result])
rotated = output.transpose(Image.ROTATE_90)
rotated.save("output.png")
