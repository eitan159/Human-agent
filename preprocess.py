import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_data(path):
    data = pd.read_excel(open(path, 'rb'))
    dummy1 = pd.get_dummies(data['Incentive conditions'])
    dummy2 = pd.get_dummies(data['DataSet'] > 3)
    new_data = pd.DataFrame({'id': data['ID'], 'rounds': data['Rounds'], 'stopping_position': data['Stopping positon'],
                             'commission_base': dummy1[1], 'best_only': dummy1[2], 'flat_fee': dummy1[3],
                             'housing': dummy2[False],
                             'no_context': dummy2[True]})
    offers = data.head(10).loc[:, 'Offer1':'Offer20'].values.tolist()

    return new_data, offers

def normalized_offers(offers):
    offers = np.array(offers).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(offers)
    offers = scaler.transform(offers)
    return offers.reshape(10, 20)


def create_data_for_LSTM_V2(data):
    round_per_id = []
    Y = []
    data = data.values
    for i in range(0, len(data), 10):
        person_rounds = []
        for j in range(i, i + 10):
            data[j][2] -= 1
            if j == i + 10 - 1:
                Y.append(data[j][2])
                data[j][2] = -1
            person_rounds.append(data[j])
        round_per_id.append(person_rounds)
    return round_per_id, Y
