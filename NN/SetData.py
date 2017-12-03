
import numpy as np

class SetData:

    def __init__(self, filename):

        self._Xtrain = []
        self._Ytrain = []
        self._Xtest = []
        self._Ytest = []

        i = 0
        for line in open(filename):
            each_data_point = line.rstrip().split(",")
            #pop id
            each_data_point.pop(0)
            if i % 4 == 0:
                #pop label
                label = each_data_point.pop(0)
                label = list(map(SetData.normalize,label))
                label = label[0]
                self._Ytest.append(label)
                self._Xtest.append(each_data_point)
            else:
                #pop label
                label = each_data_point.pop(0)
                label = list(map(SetData.normalize, label))
                label = label[0]
                self._Ytrain.append(label)
                self._Xtrain.append((each_data_point))
            i+=1
            # temp = str(line)
            # each_line = temp.rstrip().split(",")
            # each_line = each_line[1:]
            # label = each_line[0]
            # label = list(map(SetData.normalize,label))
            # label = label[0]
            # entry = each_line[1:]
            # entry = list(map(float,list(entry)))
            # self._Xtrain.append(entry)
            # self._Ytrain.append(label)
        self._Xtrain = np.array(self._Xtrain).astype(np.float)
        self._Ytrain = np.array(self._Ytrain).astype(int)
        self._Xtest = np.array(self._Xtest).astype(np.float)
        self._Ytest = np.array(self._Ytest).astype(int)


    @staticmethod
    def normalize(char):
        return [1, 0] if char == '1' else [0, 1]

def main():

    my = SetData("my.csv")
    print(my._Xtrain)
    print(my._Ytrain)



if __name__ == "__main__": main()