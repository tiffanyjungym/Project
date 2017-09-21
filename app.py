from flask import Flask
from flask import request
from flask import render_template


import pandas as pd
from sklearn.neighbors import KDTree
from lifelines import KaplanMeierFitter

from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

n_neighbors = 300

median_all = 10.797260274

patient_df = pd.read_csv('patient_df.txt')
kdtree_data = patient_df.loc[:,['TAGLN2','RARRES3','C6orf141']].as_matrix()
print(kdtree_data.shape)
tree = KDTree(kdtree_data)

def make_plot(survival_function):
    data = survival_function.reset_index().values
    p1 = figure(x_axis_type='auto', title="Kaplan-Meier plot",x_range=(0,15),y_range=(0.2,1),plot_width=400, plot_height=400)
    p1.grid.grid_line_alpha=0.3
    p1.xaxis.axis_label = 'Years'
    p1.yaxis.axis_label = 'Survival probability'

    p1.line(data[:,0], data[:,1], color='#FB9A99')
    p1.legend.location = "top_left"

    return p1

def obtain_stats(patient_ID):
    
    print(patient_df.loc[patient_df['Patient_ID']==patient_ID,['TAGLN2','RARRES3','C6orf141']])
    example_patient = patient_df.loc[patient_df['Patient_ID']==patient_ID,['TAGLN2','RARRES3','C6orf141']].as_matrix()
    example_patient = example_patient.reshape(1,-1)
    print(example_patient.shape)
    (distance,neigbor_indicies) = tree.query(example_patient,n_neighbors)

    kmf = KaplanMeierFitter()
    kmf.fit(patient_df.loc[neigbor_indicies[0],'Days']/365, 
                        patient_df.loc[neigbor_indicies[0],'Vitality'], label=' ')

    survival_function = kmf.survival_function_
    median = kmf.median_
    three_year = survival_function.loc[min(survival_function.index, key=lambda x:abs(x-3))].as_matrix()[0]
    five_year = survival_function.loc[min(survival_function.index, key=lambda x:abs(x-5))].as_matrix()[0]
    p = make_plot(survival_function)
    
    if median<median_all:
        treatment_type='AGRESSIVE'
    else:
        treatment_type='STANDARD'
        
    
        
    return median,three_year,five_year,treatment_type,p


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("my-form.html")

@app.route('/', methods=['POST'])
def my_form_post():

    patient_ID = request.form['text']
    median,three_year,five_year,treatment_type,figure = obtain_stats(patient_ID)
    fig_script, fig_div = components(figure)
    return render_template('patient.html', median=median,three_year=three_year,five_year=five_year,treatment_type=treatment_type,
                        dataframe=patient_df.loc[patient_df['Patient_ID']==patient_ID].to_html(classes='table'),fig_script=fig_script, fig_div=fig_div)

if __name__ == '__main__':
    app.run() 
