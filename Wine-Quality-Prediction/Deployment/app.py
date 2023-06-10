from flask import Flask, render_template

app = Flask(__name__)

# Main Code:
@app.route('/')
def base():
    return render_template('home.html')    # Reads 'home.html' from templates folder

@app.route('/contact')
def contact():
    return 'Contact Page'

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/cart')
def cart():
    return 'Cart Page'



app.run(debug=True)   # debug=True will auto refresh the url whenever there is an update in the code