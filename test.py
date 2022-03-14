from re import S
from gesture import classify, identify


rotatesample1 = [149, 125, 118, 102, 108, 101, 101, 97, 109, 115]
rotatesample2 = [255, 154, 150, 140, 145, 255, 139, 132, 140, 133]

zoomsample1 = [130, 122, 116, 122, 135, 138, 138, 125, 128, 130]
zoomsample2 = [115, 113, 119, 124, 137, 138, 136, 121, 122, 111]

print(classify(zoomsample1, zoomsample2))

'''
print(classify(samplerotateright1, samplerotateright2))
print(classify(samplerotateleft1, samplerotateleft2))
print(classify(samplezoomout1, samplezoomout2))
print(classify(samplezoomin1, samplezoomin2))
'''

'''
samplerotateright1 = [255, 255, 255, 255, 10, 255, 255, 255, 255, 255]
samplerotateright2 = [255, 255, 255, 255, 8, 255, 255, 255, 255, 255]

samplerotateleft1 = [255, 255, 255, 255, 8, 255, 255, 255, 255, 255]
samplerotateleft2 = [255, 255, 255, 255, 10, 255, 255, 255, 255, 255]

samplezoomout1 = [255, 255, 10, 255, 255, 255, 255, 100, 255, 255]
samplezoomout2 = [255, 255, 255, 255, 8, 7, 255, 255, 255, 255]

samplezoomin1 = [255, 255, 255, 255, 8, 7, 255, 255, 255, 255]
samplezoomin2 = [255, 255, 10, 255, 255, 255, 255, 100, 255, 255]
'''
