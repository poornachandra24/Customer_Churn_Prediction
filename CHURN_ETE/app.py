import pandas as pd 
import xgboost as xgb
from flask import Flask, request, render_template

app = Flask("__name__", template_folder='templates')

# Load your original training data (Assuming you have it in 'customer_churn_large_dataset.xlsx')
path = 'customer_churn_large_dataset.xlsx'
df_1 = pd.read_excel(path)

# Define the column order and names from your training data
training_data_columns = [ "Age", "Gender","Location",'Subscription_Length_Months', "Monthly_Bill", "Total_Usage_GB", "Total_Bill"]#,'Subscription_Length_Bins',"senior_citizen", "middle_aged", "young_adults"]

@app.route("/")
def loadPage():
    return render_template('index.html', query="")

@app.route("/", methods=['POST'])
def predict():
    inputQuery1 = int(request.form['query1'])
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = int(request.form['query4'])
    inputQuery5 = float(request.form['query5'])
    inputQuery6 = int(request.form['query6'])

    # Load the pre-trained XGBoost model
    model = xgb.Booster()
    model.load_model('xgb_model.json')

    # Create a new DataFrame with user input
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6]]
    new_df = pd.DataFrame(data, columns = ['Age','Gender','Location','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB'])
    

    # Concatenate user input with the original data
    df2 = pd.concat([df_1, new_df], ignore_index=True)


    # Ensure that the column order and names match the training data
    df2['Total_Bill']=df2['Monthly_Bill']*df2['Subscription_Length_Months']
    ####
    labels = ["{0} - {1}".format(i, i + 5) for i in range(1, 25, 6)]
    df2['Subscription_Length_Bins'] = pd.cut(df2.Subscription_Length_Months.astype(int), range(1, 30, 6), right=False, labels=labels)
    ####
    df2['senior_citizen']=(df2['Age']>=60).astype(int)
    df2['middle_aged']=((df2['Age']<60) & (df2['Age']>=40)).astype(int)
    df2['young_adults']=((df2['Age']>=18) & (df2['Age']<40)).astype(int)


    df2.drop(columns= ['CustomerID','Name','Subscription_Length_Months'], axis=1, inplace=True)
   

    new_df_dummies = pd.get_dummies(df2[["Age", "Gender", "Location", "Monthly_Bill", "Total_Usage_GB", "Total_Bill", "Subscription_Length_Bins", "senior_citizen", "middle_aged", "young_adults"]])


    # Create a DMatrix for prediction
    new_df_dmatrix = xgb.DMatrix(new_df_dummies.tail(1))

    # Make predictions using the pre-trained XGBoost model
    single = model.predict(new_df_dmatrix)
    probablity = model.predict(new_df_dmatrix)

    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {:.2f}%".format(probablity[0] * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probablity[0] * 100)

    return render_template('index.html', output1=o1, output2=o2,
                           query1=request.form['query1'],
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'],
                           query6=request.form['query6'])

if __name__ == "__main__":
    app.run()
