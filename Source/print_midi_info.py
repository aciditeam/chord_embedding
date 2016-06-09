from mido import MidiFile, MetaMessage
import sys
import os


if __name__ == '__main__':
    directory = '../Midi/Classe/'
    f_write = open('instru', 'wb')
    for fname in os.listdir(directory):
        filepath = directory + fname
        if not os.path.isfile(filepath):
            continue
        mid = MidiFile(filepath)
        f_write.write('#' * 30 + '\n')
        f_write.write(fname)
        for i, track in enumerate(mid.tracks):
            f_write.write(track.name + '\n')
            for message in track:
                if isinstance(message, MetaMessage):
                    print(message)
        f_write.write('\n')
    f_write.close()
