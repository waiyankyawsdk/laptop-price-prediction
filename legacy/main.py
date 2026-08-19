"""
Main Flask application for the Laptop Price Prediction web app.
"""
from flask import Flask, render_template, request
import numpy as np
from . import utils

# ── Load artifacts ───────────────────────────────────────────────────────────
# These are loaded once when the app starts
model = utils._load_pickle('model.pkl')
dropdown_data = utils._load_pickle('dropdown_data.pkl')


# ── App ──────────────────────────────────────────────────────────────────────
app = Flask(__name__,
            static_folder='../static',
            template_folder='../templates')

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """
    Prediction page.
    On GET, shows the form.
    On POST, processes form data and shows the predicted price.
    """
    prediction_text = ''
    if request.method == 'POST':
        # --- Get form data ---
        company     = request.form.get('company')
        type_       = request.form.get('type')
        ram         = request.form.get('ram')
        weight      = request.form.get('weight')
        touchscreen = request.form.get('touchscreen')
        ips         = request.form.get('ips')
        ppi         = request.form.get('ppi')
        cpu         = request.form.get('cpu')
        hdd         = request.form.get('hdd')
        ssd         = request.form.get('ssd')
        gpu         = request.form.get('gpu')
        os_         = request.form.get('os')

        # --- Preprocess and predict ---
        input_df = utils.preprocess_input(
            company, type_, ram, weight, touchscreen, ips, ppi,
            cpu, hdd, ssd, gpu, os_
        )
        log_price = model.predict(input_df)[0]
        price = np.exp(log_price)

        # --- Format for display ---
        prediction_text = f"Predicted Price: $ {price:,.0f}"

    return render_template(
        'prediction.html',
        prediction_text=prediction_text,
        companies=dropdown_data['companies'],
        types=dropdown_data['types'],
        cpus=dropdown_data['cpus'],
        gpus=dropdown_data['gpus'],
        oses=dropdown_data['oses'],
    )


@app.route('/data-analysis')
def data_analysis():
    """Data analysis page with Plotly charts."""
    plots = utils.generate_plots()
    return render_template('data_analysis.html', plots=plots)


@app.route('/model-performance')
def model_performance():
    """Model performance metrics page."""
    model_metrics = utils._load_pickle('model_metrics.pkl')
    perf_plots = utils.generate_performance_plots()
    
    # Format metrics for display
    mae = f"${model_metrics['mae_price']:,.0f}"
    mse = f"{model_metrics['mse_log']:.4f}"
    r2 = f"{model_metrics['r2_score']:.4f}"
    
    return render_template('model_performance.html', 
                         mae=mae, mse=mse, r2=r2,
                         plot_actual_vs_predicted=perf_plots['plot_actual_vs_predicted'],
                         plot_algorithm_comparison=perf_plots['plot_algorithm_comparison'])


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')