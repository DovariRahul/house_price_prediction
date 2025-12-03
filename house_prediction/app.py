from flask import Flask, render_template, request
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    price_usd = None

    if request.method == "POST":
        try:
            # Get data from form
            medinc = float(request.form["medinc"])
            houseage = float(request.form["houseage"])
            averooms = float(request.form["averooms"])
            avebedrooms = float(request.form["avebedrooms"])
            population = float(request.form["population"])
            aveoccp = float(request.form["aveoccp"])
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])

            user_input = np.array(
                [[medinc, houseage, averooms, avebedrooms,
                  population, aveoccp, latitude, longitude]]
            )

            prediction = model.predict(user_input)[0]  # in 100,000s
            price_usd = prediction * 100000

        except ValueError:
            prediction = "Invalid input. Please enter numeric values."
            price_usd = None

    return render_template(
        "index.html",
        prediction=prediction,
        price_usd=price_usd,
        image_url="/static/house.jpg"
    )


if __name__ == "__main__":
    app.run(debug=True)
