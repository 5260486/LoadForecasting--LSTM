import pandas as pd
import numpy as np
class ReadData():
    """
        input: file path, time frame, sheetname
        output: dateset
    """
    def __init__(self,filepath,sheetname):
        self.filepath=filepath
        self.sheetname=sheetname

    # Get the all data.
    def GetDataset(self):
        dataset =pd.read_excel(self.filepath, header=0, index_col=0,sheet_name=self.sheetname)
        return dataset

    # Extract year or month data used for predicting.
    def GetDataset(self,time_lower,time_upper):
        dataset =pd.read_excel(self.filepath, header=0, index_col=0,sheet_name=self.sheetname)
        dataset=dataset.loc[(dataset.index>=time_lower) & (dataset.index< time_upper)]
        return dataset

    # Transform  data to time and load two columns.
    def FormatData(self,newcolumnNum,time_lower,time_upper):
        data=pd.DataFrame()
        time=[]
        if newcolumnNum==2:
            data= pd.DataFrame(data, columns=['Time', 'Load'])
        elif newcolumnNum>2:
            data= pd.DataFrame(data, columns=['Time', 'Load','Feature'])

        if time_lower!=0:
            dataset=ReadData.GetDataset(self,time_lower,time_upper)

        for i in range(dataset.shape[0]):  #rows
            if i==0:
                load=dataset.iloc[i].values
            else:
                array=dataset.iloc[i].values
                load=np.concatenate((load,array),axis=0)
            for j in range(dataset.shape[1]):   #columns
                time.append(str(dataset.index[i])+dataset.columns[j])

        data['Load']=load
        data['Time']=time
        return data
