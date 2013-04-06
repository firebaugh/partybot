i = open("adele-vs-katy/someone-that-got-away-label", "r")
sequence = []
songs = []
for line in i.readlines():
    w = line.split(" ")
    songs.append(w[1])
    for j in range(0,int(w[3])-int(w[2])):
        sequence.append(0)
    sequence.append(1)
print([sequence, songs])
