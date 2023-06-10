from flask import Flask, render_template, request
import pickle
import sklearn

app = Flask(__name__)

# Main Code:
@app.route('/')
def base():
    return render_template('home.html')    # Reads 'home.html' from templates folder

@app.route('/predict', methods=['post'])
def predict():
    # Desired Inputs: 'volatile_acidity', 'residual_sugar', 'chlorides', 'total_sulfur_dioxide', 'ph', 'sulphates', 'alcohol'
    # Get the inputs from home.html
    volatile_acidity = request.form.get('volatile_acidity')
    residual_sugar = request.form.get('residual_sugar')
    chlorides = request.form.get('chlorides')
    total_sulfur_dioxide = request.form.get('total_sulfur_dioxide')
    ph = request.form.get('ph')
    sulphates = request.form.get('sulphates')
    alcohol = request.form.get('alcohol')

    print("volatile_acidity: ", volatile_acidity)
    print("residual_sugar: ", residual_sugar)
    print("chlorides: ", chlorides)
    print("total_sulfur_dioxide: ", total_sulfur_dioxide)
    print("ph: ", ph)
    print("sulphates: ", sulphates)
    print("alcohol: ", alcohol)

    # Call the Model
    model = pickle.load(open('wine_quality_prediction_80.pkl', 'rb'))
    output = model.predict([[volatile_acidity, residual_sugar, chlorides, total_sulfur_dioxide, ph, sulphates, alcohol]])
    print(output)
    data = "The quality of wine on a scale of 1-10 is: " + str(output[0])

    return render_template('predict.html', data = data)


# @app.route('/contact')
# def contact():
#     return 'Contact Page'

# @app.route('/predict')
# def predict():
#     return render_template('predict.html')

# @app.route('/cart')
# def cart():
#     return 'Cart Page'



app.run(debug=True)   # debug=True will auto refresh the url whenever there is an update in the code