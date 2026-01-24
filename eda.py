import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('eda')

class EDA:
    try:
        df = pd.read_csv('/content/Churn_Updated.csv')
        df.sample(10)

        """**1.Churn Distribution**"""

        x = (df['Churn'].value_counts(normalize=True).mul(100).plot(kind='bar', figsize=(5,4)))

        plt.xlabel('Churn')
        plt.ylabel('Percentage')
        plt.title('Churn Distribution (%)')

        for container in x.containers:
            x.bar_label(container, fmt='%.1f%%')

        plt.savefig('Churn_Distribution.png')
        plt.show()

        """**2.Gender-wise Churn Distribution**"""

        churn_gender = df[df['Churn'] == 'Yes']['gender'].value_counts(normalize=True) * 100
        x = churn_gender.plot(kind='bar', figsize=(5,4))

        plt.xlabel('Gender')
        plt.ylabel('Percentage')
        plt.title('Gender-wise Churn Percentage')

        for container in x.containers:
            x.bar_label(container, fmt='%.1f%%')

        plt.savefig('Gender_Churn.png')
        plt.show()

        """**3.Churned Customers by Gender & Senior Citizen**"""

        churn_df = df[df['Churn'] == 'Yes']
        churn_gender_senior = (pd.crosstab(churn_df['gender'], churn_df['SeniorCitizen'], normalize='index') * 100)
        churn_gender_senior.columns = ['Non-Senior', 'Senior']
        x = churn_gender_senior.plot(kind='bar', figsize=(5,4))
        plt.xlabel('Gender')
        plt.ylabel('Percentage')
        plt.title('Churned Customers by Gender and Senior Citizen Status')
        plt.legend(title='Senior Citizen')
        for container in x.containers:
          x.bar_label(container, fmt='%.1f%%')
        plt.savefig('Churn_Gender_SeniorCitizen.png')
        plt.show()

        """**4.Internet Service Usage by Gender**"""

        gender_internet_pct = (pd.crosstab(df['gender'],df['InternetService'],normalize='index') * 100)

        plt.figure(figsize=(5,2))
        ax = gender_internet_pct.plot(kind='bar')

        plt.title('Internet Service Usage by Gender (%)')
        plt.xlabel('Gender')
        plt.ylabel('Percentage')
        plt.legend(title='Internet Service')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')

        plt.tight_layout()
        plt.savefig('Gender_vs_InternetService.png')
        plt.show()

        """**5.Phone Service Usage by Gender & Senior Citizen (Churned Customers)**"""

        churn_df = df[df['Churn'] == 'Yes']

        # Create combined Gender + SeniorCitizen column
        churn_df['Gender_Senior'] = (churn_df['gender'] + '-' + churn_df['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'}))

        phone_gender_senior_pct = (pd.crosstab(churn_df['Gender_Senior'],churn_df['PhoneService'],normalize='index') * 100)

        plt.figure(figsize=(7,4))
        x = phone_gender_senior_pct.plot(kind='bar')

        plt.title('Phone Service Usage (%) among Churned Customers\nby Gender & Senior Citizen')
        plt.xlabel('Gender & Senior Citizen')
        plt.ylabel('Percentage')
        plt.legend(title='Phone Service')

        for container in x.containers:
            x.bar_label(container, fmt='%.1f%%')

        plt.tight_layout()
        plt.savefig('PhoneService_Gender_Senior_Churn.png')
        plt.show()

        """**6.Multiple Lines Usage by Gender & Senior Citizen**"""

        df['Gender_Senior'] = (df['gender'] + '-' + df['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'}))

        multiline_pct = (pd.crosstab( df['Gender_Senior'],df['MultipleLines'], normalize='index') * 100)

        plt.figure(figsize=(8,4))
        x = multiline_pct.plot(kind='bar')

        plt.title('Multiple Lines Usage by Gender & Senior Citizen')
        plt.xlabel('Gender & Senior Citizen')
        plt.ylabel('Percentage')
        plt.legend(title='Multiple Lines')

        # Add percentage labels
        for container in x.containers:
            x.bar_label(container, fmt='%.1f%%')

        plt.tight_layout()
        plt.savefig('MultipleLines_Gender_Senior.png')
        plt.show()

        """**7.MultipleLines by SIM Operator**"""

        multilines_sim_pct = (pd.crosstab(df['SIM'],df['MultipleLines'],normalize='index') * 100)

        plt.figure(figsize=(5,4))
        x = multilines_sim_pct.plot(kind='bar')

        plt.title('Multiple Lines Usage by SIM Operator')
        plt.xlabel('SIM Operator')
        plt.ylabel('Percentage')
        plt.legend(title='Multiple Lines')

        # Add percentage labels
        for container in x.containers:
            x.bar_label(container, fmt='%.1f%%')

        plt.tight_layout()
        plt.savefig('MultipleLines_by_SIM.png')
        plt.show()

        """**8.Multiple Lines Usage by SIM Operator & Gender**"""

        df['SIM_Gender'] = df['SIM'] + '-' + df['gender']

        multilines_sim_gender_pct = (pd.crosstab(df['SIM_Gender'],df['MultipleLines'],normalize='index') * 100)

        plt.figure(figsize=(5,3))
        x = multilines_sim_gender_pct.plot(kind='bar')

        plt.title('Multiple Lines Usage by SIM Operator & Gender')
        plt.xlabel('SIM - Gender')
        plt.ylabel('Percentage')
        plt.legend(title='Multiple Lines')

        # Add percentage labels
        for container in x.containers:
            x.bar_label(container, fmt='%.1f%%')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('MultipleLines_SIM_Gender.png')
        plt.show()

        """**9.Multiple Lines Usage by SIM, Gender & Senior Citizen**"""

        df['SIM_Gender_Senior'] = (df['SIM'] + '-' + df['gender'] + '-' + df['SeniorCitizen'].map({0:'Non-Senior', 1:'Senior'}))

        multilines_pct = pd.crosstab(df['SIM_Gender_Senior'],df['MultipleLines'],normalize='index') * 100

        # Plot line chart
        plt.figure(figsize=(15,5))
        for col in multilines_pct.columns:
            plt.plot(multilines_pct.index,multilines_pct[col],marker='o',label=col)

        plt.title('Multiple Lines Usage by SIM, Gender & Senior Citizen')
        plt.xlabel('SIM - Gender - Senior')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Multiple Lines')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('MultipleLines_SIM_Gender_Senior.png')
        plt.show()

        ml_df = df[df['MultipleLines'].isin(['Yes', 'No'])]

        # Group data
        grp = (
            ml_df
            .groupby(['SIM', 'gender', 'SeniorCitizen', 'MultipleLines'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        labels = [
            f"{row['SIM']}, {row['gender']}, SC:{'Yes' if row['SeniorCitizen']==1 else 'No'}"
            for _, row in grp.iterrows()
        ]

        x = np.arange(len(labels))
        width = 0.35

        ml_no = grp['No']
        ml_yes = grp['Yes']

        # Plot
        plt.figure(figsize=(16,6))
        plt.bar(x - width/2, ml_no, width, label='No Multiple Lines')
        plt.bar(x + width/2, ml_yes, width, label='Multiple Lines')

        # Value labels
        for i in range(len(x)):
            plt.text(x[i] - width/2, ml_no[i], ml_no[i],
                     ha='center', va='bottom', fontsize=9)

            plt.text(x[i] + width/2, ml_yes[i], ml_yes[i],
                     ha='center', va='bottom', fontsize=9)

        # Formatting
        plt.xticks(x, labels, rotation=30)
        plt.xlabel("SIM Operator / Gender / Senior Citizen")
        plt.ylabel("Customer Count")
        plt.title("Multiple Lines Usage by SIM Operator, Gender & Senior Citizen")
        plt.legend()
        plt.tight_layout()
        plt.show()

        """**10.Total Customers by Internet Service**"""

        internet_totals_pct = df['InternetService'].value_counts(normalize=True) * 100
        print(internet_totals_pct)

        plt.figure(figsize=(5,5))
        ax = internet_totals_pct.plot(kind='bar',color=['r','g','b'])

        plt.title('Total Customers by Internet Service')
        plt.xlabel('Internet Service')
        plt.ylabel('Percentage')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')

        plt.tight_layout()
        plt.savefig('Total_InternetService.png')
        plt.show()

        """**11.Internet Service Usage by SIM Operator**"""

        internet_sim_counts = pd.crosstab(df['InternetService'], df['SIM'])
        print(internet_sim_counts)

        internet_sim_filtered = internet_sim_counts.loc[['DSL','Fiber optic','No']]

        plt.figure(figsize=(6,4))
        ax = internet_sim_filtered.plot(kind='bar')

        plt.title('Internet Service Usage by SIM Operator')
        plt.xlabel('Internet Service')
        plt.ylabel('Count')
        plt.legend(title='SIM')

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container)

        plt.tight_layout()
        plt.savefig('InternetService_by_SIM.png')
        plt.show()

        """**12.Internet Service Usage by SIM Operator & Gender**"""

        df_int = df[df['InternetService'] != 'No']

        # Group data
        grp = (
            df_int
            .groupby(['SIM', 'gender', 'InternetService'])
            .size()
            .unstack(fill_value=0)
        )

        # Create x-axis labels
        labels = [(row[0], row[1]) for row in grp.index]
        x = np.arange(len(labels))
        width = 0.35

        # Internet service types
        dsl = grp.get('DSL', 0)
        fiber = grp.get('Fiber optic', 0)

        # Plot
        plt.figure(figsize=(14,6))
        plt.bar(x - width/2, dsl, width, label='DSL')
        plt.bar(x + width/2, fiber, width, label='Fiber Optic')

        # Formatting
        plt.xticks(x, labels, rotation=20)
        plt.xlabel("SIM Operator / Gender")
        plt.ylabel("Customer Count")
        plt.title("Internet Service Usage by SIM Operator & Gender")
        plt.legend(title="Internet Service")
        plt.tight_layout()
        plt.show()

        """**13.Internet Service Usage by SIM Operator, Gender & Churn**"""

        df_int = df[df['InternetService'] != 'No']

        # Group data
        grp = (
            df_int
            .groupby(['SIM', 'gender', 'Churn'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        labels = [(row['SIM'], row['gender']) for _, row in grp.iterrows()]
        x = np.arange(len(labels))
        width = 0.35

        no_churn = grp['No']
        yes_churn = grp['Yes']
        total = no_churn + yes_churn

        # Plot
        plt.figure(figsize=(14,6))
        plt.bar(x - width/2, no_churn, width, label='No')
        plt.bar(x + width/2, yes_churn, width, label='Yes')

        # Percentage inside bars
        for i in range(len(x)):
            plt.text(x[i] - width/2, no_churn[i]/2,
                     f"{(no_churn[i]/total[i])*100:.1f}%",
                     ha='center', va='center', fontsize=9)

            plt.text(x[i] + width/2, yes_churn[i]/2,
                     f"{(yes_churn[i]/total[i])*100:.1f}%",
                     ha='center', va='center', fontsize=9)

        # Formatting
        plt.xticks(x, labels, rotation=20)
        plt.xlabel("SIM Operator / Gender")
        plt.ylabel("Customer Count")
        plt.title("Internet Service Usage by SIM Operator, Gender & Churn")
        plt.legend(title="Churn")
        plt.tight_layout()
        plt.show()

        """**14.OnlineSecurity,**
        **OnlineBackup,**
        **DeviceProtection,**
        **TechSupport,**
        **StreamingTV,**
        **StreamingMovies**
        """

        service_cols = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]

        plt.figure(figsize=(18,10))

        # Loop through each column and create subplot
        for i, col in enumerate(service_cols, 1):
            plt.subplot(2, 3, i)

            # Calculate percentage
            pct = df[col].value_counts(normalize=True) * 100

            ax = pct.plot(kind='bar', color=['g','orange','red'])

            plt.title(f'{col} Usage (%)')
            plt.xlabel('')
            plt.ylabel('Percentage')

            # Add percentage labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%')

        plt.tight_layout()
        plt.savefig('ServiceColumns.png')
        plt.show()

        service_cols = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]

        plt.figure(figsize=(18,10))

        for i, col in enumerate(service_cols, 1):
            plt.subplot(2, 3, i)

            pct = pd.crosstab(df['gender'], df[col], normalize='index') * 100

            ax = pct.plot(kind='bar', ax=plt.gca())

            plt.title(f'{col} Usage by Gender (%)')
            plt.xlabel('Gender')
            plt.ylabel('Percentage')

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%')

        plt.tight_layout()
        plt.savefig('ServiceColumns_By_Gender.png')
        plt.show()

        """**15.Contract Distribution**"""

        for sim in df['SIM'].unique():

            sim_df = df[df['SIM'] == sim]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            width = 0.35

            for i, churn in enumerate(['No', 'Yes']):
                temp = sim_df[sim_df['Churn'] == churn]

                # Group data
                grouped = (
                    temp.groupby(['Contract', 'gender'])
                        .size()
                        .unstack(fill_value=0)
                )

                x = np.arange(len(grouped))

                # Bars
                axes[i].bar(x - width/2, grouped['Male'], width, label='Male')
                axes[i].bar(x + width/2, grouped['Female'], width, label='Female')

                # Values on bars
                for j in range(len(grouped)):
                    axes[i].text(x[j] - width/2, grouped['Male'].iloc[j],
                                 grouped['Male'].iloc[j], ha='center', va='bottom', fontsize=8)
                    axes[i].text(x[j] + width/2, grouped['Female'].iloc[j],
                                 grouped['Female'].iloc[j], ha='center', va='bottom', fontsize=8)

                axes[i].set_xticks(x)
                axes[i].set_xticklabels(grouped.index, rotation=15)
                axes[i].set_title(f'Churn = {churn}')
                axes[i].set_xlabel('Contract')
                axes[i].set_ylabel('Number of Customers')
                axes[i].legend(title='Gender')

            # Main title for each SIM
            fig.suptitle(f'Contract vs Gender by Churn (SIM = {sim})')

            plt.tight_layout()
            plt.show()

        """**Contract Distribution by Gender, Senior Citizen & SIM**"""

        df['Gender_Senior_SIM'] = df['gender'] + '-' + df['SeniorCitizen'].map({0:'Non-Senior', 1:'Senior'}) + '-' + df['SIM']

        # Contract by Gender + Senior + SIM (%)
        contract_demographic_pct = pd.crosstab(df['Gender_Senior_SIM'], df['Contract'], normalize='index') * 100

        plt.figure(figsize=(15,6))
        ax = contract_demographic_pct.plot(kind='bar', ax=plt.gca())
        plt.title('Contract Distribution by Gender, Senior Citizen & SIM')
        plt.xlabel('Gender - Senior - SIM')
        plt.ylabel('Percentage')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('Contract_By_Gender_Senior_SIM.png')
        plt.show()

        """**16.Paperless Billing by Gender, SIM Operator & Churn**"""

        grp = (
            df.groupby(['SIM', 'gender', 'Churn', 'PaperlessBilling'])
              .size()
              .unstack(fill_value=0)
              .reset_index()
        )

        labels = [
            f"{row['SIM']}, {row['gender']}, {row['Churn']}"
            for _, row in grp.iterrows()
        ]

        x = np.arange(len(labels))
        width = 0.35

        paper_no = grp['No']
        paper_yes = grp['Yes']
        total = paper_no + paper_yes

        # Plot
        plt.figure(figsize=(16,6))
        plt.bar(x - width/2, paper_no, width, label='Paper Billing')
        plt.bar(x + width/2, paper_yes, width, label='Paperless Billing')

        # Percentage labels
        for i in range(len(x)):
            plt.text(x[i] - width/2, paper_no[i]/2,
                     f"{(paper_no[i]/total[i])*100:.1f}%",
                     ha='center', va='center', fontsize=9)

            plt.text(x[i] + width/2, paper_yes[i]/2,
                     f"{(paper_yes[i]/total[i])*100:.1f}%",
                     ha='center', va='center', fontsize=9)

        # Formatting
        plt.xticks(x, labels, rotation=30)
        plt.xlabel("SIM Operator / Gender / Churn")
        plt.ylabel("Customer Count")
        plt.title("Paperless Billing by Gender, SIM Operator & Churn")
        plt.legend()
        plt.tight_layout()
        plt.show()

        """**17.Payment Method by Gender & Churn**"""

        grp = (
            df.groupby(['PaymentMethod', 'gender', 'Churn'])
              .size()
              .unstack(fill_value=0)
              .reset_index()
        )

        labels = [
            f"{row['PaymentMethod']}, {row['gender']}"
            for _, row in grp.iterrows()
        ]

        x = np.arange(len(labels))
        width = 0.35

        no_churn = grp['No']
        yes_churn = grp['Yes']
        total = no_churn + yes_churn

        # Plot
        plt.figure(figsize=(16,6))
        plt.bar(x - width/2, no_churn, width, label='No Churn')
        plt.bar(x + width/2, yes_churn, width, label='Churn')

        # Percentage labels
        for i in range(len(x)):
            plt.text(x[i] - width/2, no_churn[i]/2,
                     f"{(no_churn[i]/total[i])*100:.1f}%",
                     ha='center', va='center', fontsize=9)

            plt.text(x[i] + width/2, yes_churn[i]/2,
                     f"{(yes_churn[i]/total[i])*100:.1f}%",
                     ha='center', va='center', fontsize=9)

        # Formatting
        plt.xticks(x, labels, rotation=30)
        plt.xlabel("Payment Method / Gender")
        plt.ylabel("Customer Count")
        plt.title("Payment Method by Gender & Churn")
        plt.legend()
        plt.tight_layout()
        plt.show()

        """**18.Quarterly Tenure Analysis with SIM**"""

        for sim in df['SIM'].unique():
            sim_df = df[df['SIM'] == sim]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            width = 0.45

            for i, churn in enumerate(['No', 'Yes']):
                temp = sim_df[sim_df['Churn'] == churn]

                grouped = (
                    temp.groupby([((temp['tenure'] - 1) // 3 + 1), 'gender'])
                        .size()
                        .unstack(fill_value=0)
                )

                x = np.arange(len(grouped))

                axes[i].bar(x - width/2, grouped['Male'], width, label='Male')
                axes[i].bar(x + width/2, grouped['Female'], width, label='Female')

                for j in range(len(grouped)):
                    axes[i].text(x[j] - width/2, grouped['Male'].iloc[j],
                                 grouped['Male'].iloc[j], ha='center', va='bottom', fontsize=8)
                    axes[i].text(x[j] + width/2, grouped['Female'].iloc[j],
                                 grouped['Female'].iloc[j], ha='center', va='bottom', fontsize=8)

                axes[i].set_xticks(x)
                axes[i].set_xticklabels([int(q) for q in grouped.index])  # 🔹 numbers only
                axes[i].set_title(f'Churn = {churn}')
                axes[i].set_xlabel('Tenure (Quarter Number)')
                axes[i].set_ylabel('Customers')
                axes[i].legend(title='Gender')

            fig.suptitle(f'Quarterly Tenure Analysis | SIM = {sim}')
            plt.tight_layout()
            plt.show()

        """**19.Monthly Charges vs SIM Operator by Churn**"""

        avg_charges = (
            df.groupby(['SIM', 'Churn'])['MonthlyCharges']
              .mean()
              .unstack()
        )

        x = np.arange(len(avg_charges.index))
        width = 0.35

        no_churn = avg_charges['No']
        yes_churn = avg_charges['Yes']

        # Plot
        plt.figure(figsize=(10,6))
        bars_no = plt.bar(x - width/2, no_churn, width, label='No Churn')
        bars_yes = plt.bar(x + width/2, yes_churn, width, label='Churn')

        # Percentage labels
        for i in range(len(x)):
            total = no_churn[i] + yes_churn[i]

            no_pct = (no_churn[i] / total) * 100
            yes_pct = (yes_churn[i] / total) * 100

            plt.text(x[i] - width/2, no_churn[i],
                     f"{no_pct:.1f}%",
                     ha='center', va='bottom', fontsize=9)

            plt.text(x[i] + width/2, yes_churn[i],
                     f"{yes_pct:.1f}%",
                     ha='center', va='bottom', fontsize=9)

        # Formatting
        plt.xticks(x, avg_charges.index)
        plt.xlabel("SIM Operator")
        plt.ylabel("Average Monthly Charges")
        plt.title("Average Monthly Charges by SIM Operator & Churn (with %)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        """**20.Region vs Gender with Senior Citizen**"""

        df_non_senior = df[df['SeniorCitizen'] == 0]
        df_senior = df[df['SeniorCitizen'] == 1]

        # Group data
        grp_non = df_non_senior.groupby(['Region', 'gender']).size().unstack(fill_value=0)
        grp_sen = df_senior.groupby(['Region', 'gender']).size().unstack(fill_value=0)

        regions = grp_non.index
        x = np.arange(len(regions))
        width = 0.35

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=True)

        # ---- Non-Senior Citizens ----
        axes[0].bar(x - width/2, grp_non['Male'], width, label='Male')
        axes[0].bar(x + width/2, grp_non['Female'], width, label='Female')
        axes[0].set_title("Region vs Gender (Non-Senior Citizens)")
        axes[0].set_xlabel("Region")
        axes[0].set_ylabel("Customer Count")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(regions, rotation=20)
        axes[0].legend()

        # ---- Senior Citizens ----
        axes[1].bar(x - width/2, grp_sen['Male'], width, label='Male')
        axes[1].bar(x + width/2, grp_sen['Female'], width, label='Female')
        axes[1].set_title("Region vs Gender (Senior Citizens)")
        axes[1].set_xlabel("Region")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(regions, rotation=20)
        axes[1].legend()

        plt.suptitle("Region vs Gender with Senior Citizen")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
