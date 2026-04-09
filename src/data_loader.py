# import pandas as pd

# def load_data(path):

#     df = pd.read_csv(path)
#     print(df.columns)
#     # Combine date + time
#     df['datetime'] = pd.to_datetime(
#         df['transaction_date'] + ' ' + df['transaction_time']
#     )

#     # Revenue
#     df['revenue'] = df['transaction_qty'] * df['unit_price']

#     return df

import pandas as pd

def load_data(path):
    path=r"C:\Users\Shramanth P Acharya\OneDrive\Documents\Project\Unified Mentor Private\coffee_demand_forecasting - Copy\data\cofee_sales.csv"
    # ✅ Use dynamic path
    df = pd.read_csv(path)

    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)

    # Revenue
    if 'revenue' not in df.columns:
        df['revenue'] = df['transaction_qty'] * df['unit_price']

    return df