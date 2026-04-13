from PIL import Image
img = Image.frombytes('L', (28, 28), open('data/inputs/image_0.ubyte', 'rb').read())
img.save('digit.png')