import gesture

samplerotateright1 = [255, 255, 255, 255, 10, 255, 255, 255, 255, 255]
samplerotateright2 = [255, 255, 255, 255, 8, 255, 255, 255, 255, 255]

samplerotateleft1 = [255, 255, 255, 255, 8, 255, 255, 255, 255, 255]
samplerotateleft2 = [255, 255, 255, 255, 10, 255, 255, 255, 255, 255]

samplezoomout1 = [255, 255, 10, 255, 255, 255, 255, 100, 255, 255]
samplezoomout2 = [255, 255, 255, 255, 8, 7, 255, 255, 255, 255]

samplezoomin1 = [255, 255, 255, 255, 8, 7, 255, 255, 255, 255]
samplezoomin2 = [255, 255, 10, 255, 255, 255, 255, 100, 255, 255]

print(gesture.classify(samplerotateright1, samplerotateright2))
print(gesture.classify(samplerotateleft1, samplerotateleft2))
print(gesture.classify(samplezoomout1, samplezoomout2))
print(gesture.classify(samplezoomin1, samplezoomin2))
