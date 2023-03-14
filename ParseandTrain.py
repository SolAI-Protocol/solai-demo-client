import zipfile
import pandas as pd
from sklearn.impute import KNNImputer
import time
import re
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import torch
from torch import nn,optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

SOLUSD = pd.read_csv('SOLUSD.csv')
SOLUSD['Date']= pd.to_datetime(SOLUSD['Date'])
SOLUSD.dropna(inplace=True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3) #연산 마다 30%의 노드를 랜덤하게 없앤다
        self.bn1 = torch.nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.fc1(x)) #활성화 함수 적용 
        x = self.bn1(x)
        x = F.relu(self.fc2(x)) 
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x

def parse(directory):
    start = time.time()
    zip = zipfile.ZipFile(directory + '/' + 'data.zip', 'r')
    files = [[zinfo.filename, zinfo.file_size] for zinfo in zip.filelist if zinfo.filename.endswith('xml')] # xml로 끝나는 파일들
    files.sort(key=lambda x:x[1],reverse=True) # 제일 큰 file이 export.xml
    export_file = files[0][0]
    print("select file : ", export_file)
    
    zipinfos = zip.infolist()
    for zipinfo in zipinfos:
        if zipinfo.filename==export_file:
            zipinfo.filename = directory + '/apple_health_export/export.xml'
            zip.extract(zipinfo)

    healthlog = open(directory + '/apple_health_export/export.xml',"r", encoding='UTF8')
    data=[]
    FMT = '%Y-%m-%d %H:%M:%S'

 
    # loop through export.eml
    for line in healthlog:
        #find record types
        if re.search(r"<Record type=", line):
            recordtype = re.search(r"<Record type=\"\S+\"",line)
            recordtypeval = recordtype.group()[14:-1]

            # # get source of value
            # sourceName =re.search(r"model:[^,]*,",line)
            # if sourceName is None:
            #     sourceNameval = "No Model"
            # else:
            #     sourceNameval = sourceName.group()[6:-1]

            starttime = datetime.strptime(re.search(r"startDate\S\S\d+\-\d+\-\d+\s+\d+\:\d+\:\d+",line).group()[11:], FMT) 
            endtime = datetime.strptime(re.search(r"endDate\S\S\d+\-\d+\-\d+\s+\d+\:\d+\:\d+",line).group()[9:], FMT) 
            tdelta = endtime-starttime

            # Get value of record type 
            healthdata = re.search(r"value\S\S\w+",line) 
            if healthdata is None:
                healthdata = "No Val"
            else: 
                if 'IdentifierSleepAnalysis' in recordtypeval:
                    healthdataval = "0000000" + str(tdelta)[:1]
                else:
                    healthdataval = healthdata.group()
            try:
                healthdataval = float(healthdataval[7:])
                healthdataval /= tdelta
            except:
                pass

            # Output results to file
            data.append({'datetime':endtime,recordtypeval:healthdataval})
        # if re.search(r"<Workout", line):
        #     break

    #Close files
    healthlog.close()
    data_df=pd.DataFrame(data)

    end = time.time()
    print(f"{end - start:.5f}sec parsing done")
    return data_df

def preprocess(data_df):
    start = time.time()
    data_df.set_index('datetime',inplace=True)
    data_df = data_df.resample(rule='D').mean() #일별 평균값
    data_df.reset_index(inplace=True)
    data_df = pd.merge(SOLUSD,data_df,how='left',left_on='Date',right_on='datetime')
    data_df.set_index('datetime',inplace=True)
    data_df.drop('Date',axis=1,inplace=True)
    data_df = data_df[["ER","HKQuantityTypeIdentifierStepCount","HKQuantityTypeIdentifierFlightsClimbed","HKQuantityTypeIdentifierBasalEnergyBurned","HKQuantityTypeIdentifierWalkingStepLength","HKQuantityTypeIdentifierWalkingSpeed","HKQuantityTypeIdentifierHeadphoneAudioExposure","HKQuantityTypeIdentifierActiveEnergyBurned","HKQuantityTypeIdentifierHeartRate","HKQuantityTypeIdentifierAppleStandTime","HKQuantityTypeIdentifierWalkingHeartRateAverage","HKQuantityTypeIdentifierRestingHeartRate","HKQuantityTypeIdentifierHeartRateVariabilitySDNN"]]
    noner_df = data_df.drop('ER',axis=1)
    er_df = data_df[['ER']]
	# 스케일링
    scaler = MinMaxScaler()
    scaler.fit(noner_df)
    scaled_data = scaler.transform(noner_df)

    erscaler = MaxAbsScaler()
    erscaler.fit(er_df)
    erscaled_data = erscaler.transform(er_df)
 
    # 데이터 변환
    imputer = KNNImputer(n_neighbors=5)
    scaled_data = imputer.fit_transform(scaled_data)
    # rescale_data = mMscaler.inverse_transform(mMscaled_data)
    scaled_data = pd.DataFrame(scaled_data, index=noner_df.index,columns=noner_df.columns)
    scaled_data['ER'] = erscaled_data
    end = time.time()
    print(f"{end - start:.5f}sec preprocessing done")
    return scaled_data


def dataset(data):
    start = time.time()
    X = data.drop('ER', axis=1).to_numpy()
    y = data['ER'].to_numpy().reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    ds_train = TensorDataset(X_train, y_train)
    ds_test = TensorDataset(X_test, y_test)

    loader_train = DataLoader(ds_train, batch_size=32, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=32, shuffle=False)
    end = time.time()
    print(f"{end - start:.5f}sec dataset done")

    return loader_train,loader_test

def train(trainset,validset,model,epoch):
    start = time.time()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)

    train_loss = []
    test_loss = []

    for _ in range(epoch):
        model.train()  # 신경망을 학습 모드로 전환
        running_loss =0.0 
        for inputs, values in trainset:
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, values) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss/len(trainset))

        test_loss.append(test(validset,model))

    plt.subplot(2, 1, 1)
    plt.plot(train_loss,color='b')
    plt.ylabel('train')
    plt.title('loss')
    plt.subplot(2, 1, 2)
    plt.plot(test_loss,color='r')
    plt.ylabel('test')
    plt.xlabel('epoch')
    plt.savefig('loss.png')
    plt.clf()


    end = time.time()
    print(f"{end - start:.5f}sec train done")

    return model
    
def test(dataset,model):
    model.eval() #dropout과 같은 모델 학습시에만 사용하는 기법들을 비활성화 
    predictions = torch.tensor([], dtype=torch.float) #예측값 저장을 위한 빈 텐서 
    actual = torch.tensor([], dtype=torch.float) #실제값 저장을 위한 빈 텐서 
    with torch.no_grad(): #requires_grad 비활성화 
        
        #배치 단위로 데이터를 예측하고 예측값과 실제값을 누적해서 저장 
        for inputs, values  in dataset:
            outputs = model(inputs)
            
            #0차원으로 누적한다는 의미
            predictions = torch.cat((predictions, outputs), 0)  
            actual = torch.cat((actual, values), 0)
    
    predictions = predictions.numpy()
    actual = actual.numpy()
    mse = mean_squared_error(predictions, actual)

    return mse



if __name__ == "__main__":
    start = time.time()
    parsed_data = parse('export3.zip')
    pre_data = preprocess(parsed_data)
    loader_train,loader_test = dataset(pre_data)
    model = Model()
    model = train(loader_train,loader_test,model,200)
    train_mse = test(loader_train,model)
    test_mse = test(loader_test,model)
    print('train set : ', train_mse)
    print('test set : ', test_mse)
    end = time.time()
    print(f"{end - start:.5f}sec total")

    result = open("result.txt","w")
    result.write(f"{end - start:.5f}sec, train_mse : {train_mse}, test_mse : {test_mse}")
    result.close()