import os


with open('indexes_landmarks.txt', 'w') as outfile:
    with open('indexes.txt', 'r') as infile:
        for img_path,x,y in map(lambda x: x.split(" "), infile.readlines()):
            if os.path.isdir('output_landmarks/{0}'.format(img_path)):
                print("{0} exists".format(img_path))
                outfile.write("{0} {1} {2}".format(img_path, x, y))
            else:
                print("{0} does not exist".format(img_path))
