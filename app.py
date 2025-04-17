from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define features (same as used in training)
features = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
    'four_g', 'int_memory', 'm_dep', 'mobile_wt',
    'n_cores', 'pc', 'px_height', 'px_width', 'ram',
    'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen',
    'wifi'
]

# Price range mapping
price_ranges = {
    0: "Low Cost (0-25% percentile)",
    1: "Medium Cost (25-50% percentile)",
    2: "High Cost (50-75% percentile)",
    3: "Very High Cost (75-100% percentile)"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        form_data = {
            'battery_power': float(request.form['battery_power']),
            'blue': int(request.form.get('blue', 0)),
            'clock_speed': float(request.form['clock_speed']),
            'dual_sim': int(request.form.get('dual_sim', 0)),
            'fc': float(request.form['fc']),
            'four_g': int(request.form.get('four_g', 0)),
            'int_memory': float(request.form['int_memory']),
            'm_dep': float(request.form['m_dep']),
            'mobile_wt': float(request.form['mobile_wt']),
            'n_cores': float(request.form['n_cores']),
            'pc': float(request.form['pc']),
            'px_height': float(request.form['px_height']),
            'px_width': float(request.form['px_width']),
            'ram': float(request.form['ram']),
            'sc_h': float(request.form['sc_h']),
            'sc_w': float(request.form['sc_w']),
            'talk_time': float(request.form['talk_time']),
            'three_g': int(request.form.get('three_g', 0)),
            'touch_screen': int(request.form.get('touch_screen', 0)),
            'wifi': int(request.form.get('wifi', 0))
        }
        
        # Create DataFrame from form data
        input_data = pd.DataFrame([form_data])
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_range = prediction[0]
        
        return render_template('result.html', 
                            prediction=price_ranges[predicted_range],
                            form_data=form_data)
    
    return render_template('index.html', features=features)

if __name__ == '__main__':
    app.run(debug=True)