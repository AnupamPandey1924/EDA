import nbformat as nbf 
import os

def generate_eda_notebook(data='UCI_Credit_Card.csv', location='', target='default.payment.next.month'):
    nb = nbf.v4.new_notebook()
    text = """## Exploratory Data Analysis """
    
    code_import_modules = """import os
import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn')
plt.rcParams.update({'figure.max_open_warning': 0})
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
"""
    code_javascript="""%%javascript
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
"""
    
    code_declarations = """input_data_name = '%s' 
input_data_location = '%s'
target = '%s'""" % (data, location, target)
    
    code_read_dataset = "input_data = pd.read_csv(os.path.join(input_data_location, input_data_name))"
    
    code_support_functions = """def contents(df):
    val_df      = pd.DataFrame(list(df.apply(lambda x: {'Variable'        : x.name,
                                                        'Distinct'        : x.nunique(),
                                                        'Missing'         : x.isna().sum(),
                                                        'PerMissing'      : np.round(x.isna().sum()/len(x),4)*100,
                                                        'Sample'     : x.sample(1).values[0]})))
    type_df     = pd.DataFrame(df.dtypes,columns=['class'])
    type_df     = type_df.reset_index(drop=False).rename(columns={'index':'Variable'})
    contents_df = val_df.merge(type_df)
    contents_df['Type'] = np.where(contents_df['class']=='object','Categorical',
                                  np.where(contents_df['class']=='datetime64[ns]','Date','Continuous'))
    contents_df = contents_df[['Variable','class','Distinct','Missing','PerMissing','Sample',
                               'Type']]
    return(contents_df)
    
def generate_bivariate_plots(df, target, variables):
    for variable in variables:
        df_bins        = pd.DataFrame({'bins': pd.qcut(df[variable], 10, duplicates='drop')})
        df_target      = pd.DataFrame({'target': df.loc[:,target]})
        df_count       = pd.concat([df_bins, df_target], axis=1).groupby('bins').count().reset_index()
        df_response    = pd.concat([df_bins, df_target], axis=1).groupby('bins').mean().reset_index()
        no_of_bins     = df_bins['bins'].nunique()

        df_count.columns    = ['bins', 'count']
        df_response.columns = ['bins', 'response']

        df_to_plot  = pd.concat([df_response, df_count['count']], axis=1)
        ax = df_to_plot[['bins', 'count']].plot(x = 'bins', kind='bar',  secondary_y=True, alpha=0.5, rot=60, xlim=(-0.5, no_of_bins-0.5))
        df_to_plot[['bins', 'response']].plot(x='bins', linestyle="-", title=variable, marker = 'o', rot=60, ax=ax, color='darkgreen', xlim=(-0.5, no_of_bins-0.5), figsize=(15,8))"""
    
    code_contents_report = """df_contents = contents(input_data)
df_contents"""
    
    code_bivariate = """continuous_variables = df_contents['Variable'][(df_contents['Type']=='Continuous') & (df_contents['Variable'] != target)]
generate_bivariate_plots(input_data, target, continuous_variables)"""
    
    nb['cells'] = [nbf.v4.new_markdown_cell(text),
                   nbf.v4.new_markdown_cell('### Importing modules'),
                   nbf.v4.new_code_cell(code_import_modules),
                   nbf.v4.new_code_cell(code_javascript),
                   nbf.v4.new_markdown_cell('### Support functions'),
                   nbf.v4.new_code_cell(code_support_functions),
                   nbf.v4.new_markdown_cell('### Global declarations'),
                   nbf.v4.new_code_cell(code_declarations),
                   nbf.v4.new_markdown_cell('### Loading the dataset'),
                   nbf.v4.new_code_cell(code_read_dataset),
                   nbf.v4.new_markdown_cell('### Data quality report'),
                   nbf.v4.new_code_cell(code_contents_report),
                   nbf.v4.new_markdown_cell('### Bivariates'),
                   nbf.v4.new_code_cell(code_bivariate)
                   ]
    nbf.write(nb, 'EDA.ipynb')
    
def main():
    generate_eda_notebook()
    os.system('jupyter nbconvert --execute --inplace EDA.ipynb')
    os.system('jupyter nbconvert --to HTML EDA.ipynb')
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert import PDFExporter
    
if __name__ == '__main__':
    main()