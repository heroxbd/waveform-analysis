import numpy as np
import tables
import multiprocessing as mp



'''

if __name__ == '__main__':

    evt = []
    chan = []
    pet = []
    wei = []


    for iden in np.arange(8):
        print('processing data %d' % iden)
        filename = '../output/output_' + str(iden) + '.txt'
        with open(filename, 'r') as f:
            for line in f:
                data = line.split()
                evt.append =(int(data[0]))
                chan.append(int(data[1]))
                pet.append(int(float(data[2])))
                wei.append(float(data[3]))

    evt = np.array(evt)
    chan = np.array(chan)
    pet = np.array(pet)
    wei = np.array(wei)
    
'''


class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)

# Create the output file and the group
h5file = tables.open_file("answer.h5", mode="w", title="OneTonDetector")

# Create tables
AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
answer = AnswerTable.row

# Write data
for i in range(8):
    filename = '../output/output_' + str(i) + '.txt'
    print(i)
    with open(filename, 'r') as f:
        for line in f:
            data = line.split()
            answer['EventID'] = int(data[0])
            answer['ChannelID'] = int(data[1])
            answer['PETime'] = int(float(data[2]))
            answer['Weight'] = float(data[3])
            answer.append()
AnswerTable.flush()
# Flush into the output file


h5file.close()



# class AnswerData(tables.IsDescription):
#     EventID    = tables.Int64Col(pos=0)
#     ChannelID  = tables.Int16Col(pos=1)
#     PETime     = tables.Int16Col(pos=2)
#     Weight     = tables.Float32Col(pos=3)

# # Create the output file and the group
# h5file = tables.open_file("answer.h5", mode="w", title="OneTonDetector")

# # Create tables
# AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
# answer = AnswerTable.row

# # Write data
# # for i in range(8):
# #     filename = '../output/output_' + str(i) + '.txt'
# #     print(i)
# #     with open(filename, 'r') as f:
# #         for line in f:
# #             data = line.split()
# #             answer['EventID'] = int(data[0])
# #             answer['ChannelID'] = int(data[1])
# #             answer['PETime'] = int(float(data[2]))
# #             answer['Weight'] = float(data[3])
# #             answer.append()
# # AnswerTable.flush()
# # Flush into the output file


# h5file.close()








# # 改进：阈值和偏移量