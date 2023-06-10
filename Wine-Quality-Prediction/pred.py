import pickle

model = pickle.load(open('wine_quality_prediction_80.pkl', 'rb'))
data = model.predict([[0.32, 6.4, 0.073, 13, 3.23, 0.82, 12.6]])
print(data)

data = model.predict([[0.5,7.5,0.5,20,5.5,1.5,15]])
print(data)