import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/Users/kumar/Downloads/Drug_overdose_death_rates__by_drug_type__sex__age__race__and_Hispanic_origin__United_States - Copy.csv"
df = pd.read_csv(file_path)

# Explore structure and missing values
print("Dataset Info:\n")
print(df.info())

print("\nMissing Values Before Cleaning:\n")
print(df.isna().sum()[df.isna().sum() > 0])

# Fill missing values

# Fill numeric columns with mean
numeric_columns = df.select_dtypes(include='number').columns
for col in numeric_columns:
    if df[col].isna().any():
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)

# Fill non-numeric columns with mode
non_numeric_columns = df.select_dtypes(exclude='number').columns
for col in non_numeric_columns:
    if df[col].isna().any():
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col].fillna(mode_value[0], inplace=True)

# Check missing values after cleaning
print("\nMissing Values After Cleaning:\n")
print(df.isna().sum()[df.isna().sum() > 0])

# Anomalies

# Set plot style
sns.set(style="whitegrid")

# Plot YEAR vs DEATH RATE to identify spikes/drops (anomalies)
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="YEAR", y="DEATH RATE", hue="DRUG TYPE", alpha=0.7, palette="Set2")

# Customize the plot
plt.title("Anomaly Detection: Death Rate by Year and Drug Type")
plt.xlabel("Year")
plt.ylabel("Death Rate")
plt.legend(title="Drug Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Skewness

# Calculate skewness
death_rate_skew = df["DEATH RATE"].skew()

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(df["DEATH RATE"], bins=30, kde=True, color='skyblue')

# Add text with skewness value
plt.title(f"Distribution of Death Rate (Skewness = {death_rate_skew:.2f})")
plt.xlabel("Death Rate")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()

# Correlation

# Compute correlation matrix for numeric columns
corr_matrix = df.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create heatmap using seaborn
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='gray')

# Customize plot
plt.title("Correlation Heatmap of Numeric Columns")
plt.tight_layout()

# Show plot
plt.show()

# Trends and Patterns

# Set the style
sns.set(style="whitegrid")

# Plot trends: YEAR vs DEATH RATE, grouped by DRUG TYPE
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x="YEAR", y="DEATH RATE", hue="DRUG TYPE", estimator="mean", ci=None, marker="o", palette="tab10")

# Customize the plot
plt.title("Trend of Overdose Death Rate by Drug Type Over Time")
plt.xlabel("Year")
plt.ylabel("Average Death Rate")
plt.legend(title="Drug Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Objective 1 Identify which drug type caused the most deaths

# Group by DRUG TYPE and sum the DEATH RATE
drug_death_totals = df.groupby("DRUG TYPE")["DEATH RATE"].sum().sort_values(ascending=False)

# Plot using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x=drug_death_totals.index, y=drug_death_totals.values, palette="viridis")

# Customize the plot
plt.xticks(rotation=45, ha='right')
plt.xlabel("Drug Type")
plt.ylabel("Total Death Rate")
plt.title("Total Death Rate by Drug Type in the U.S.")
plt.tight_layout()

# Show plot
plt.show()

# Conclusion
# The drug type with the highest total overdose death rate is Synthetic opioids other
# than methadone. This category significantly surpasses all others, highlighting its
# critical role in the U.S. opioid crisis. Public health strategies should prioritize
# addressing synthetic opioids due to their high lethality.

# Objective 2 Compare overdose death rates between males and females

# Filter data for only "Male" and "Female" categories in the "SEX AND RACE" column
sex_filter = df["SEX AND RACE"].isin(["Male", "Female"])
df_sex = df[sex_filter]

# Group by SEX AND RACE and calculate the mean death rate
sex_death_avg = df_sex.groupby("SEX AND RACE")["DEATH RATE"].mean().reset_index()

# Plotting
plt.figure(figsize=(8, 5))
sns.barplot(x="SEX AND RACE", y="DEATH RATE", data=sex_death_avg, palette="Set2")

# Customize plot
plt.xlabel("Sex")
plt.ylabel("Average Death Rate")
plt.title("Average Overdose Death Rate: Males vs. Females")
plt.tight_layout()

# Show plot
plt.show()

# Conclusion
# The average overdose death rate is consistently higher for males than for females.
# This indicates that men are more severely affected by drug overdoses, suggesting the
# need for gender-specific prevention and intervention programs.

# Objective 3 Find the age group with the highest overdose death rate

# Group by AGE and calculate the mean death rate
age_death_avg = df.groupby("AGE")["DEATH RATE"].mean().reset_index()

# Sort age groups by average death rate (optional)
age_death_avg = age_death_avg.sort_values(by="DEATH RATE", ascending=False)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x="AGE", y="DEATH RATE", data=age_death_avg, palette="coolwarm")

# Customize plot
plt.xticks(rotation=45, ha='right')
plt.xlabel("Age Group")
plt.ylabel("Average Death Rate")
plt.title("Average Overdose Death Rate by Age Group")
plt.tight_layout()

# Show plot
plt.show()

# Conclusion
# The 25–34 age group exhibits the highest average overdose death rate, followed closely
# by the 35–44 age group. This suggests that young to middle-aged adults are at the
# greatest risk and should be a key demographic for overdose prevention efforts.

# Objective 4 See how overdose deaths have changed from year to year

# Group by YEAR and calculate the average death rate
yearly_death_avg = df.groupby("YEAR")["DEATH RATE"].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(x="YEAR", y="DEATH RATE", data=yearly_death_avg, marker="o", linewidth=2.5)

# Customize plot
plt.xlabel("Year")
plt.ylabel("Average Death Rate")
plt.title("Trend of Overdose Death Rate Over the Years")
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()

# Conclusion
# Overdose death rates have shown a steady and significant increase over the years,
# especially after 2013. This reflects the worsening drug epidemic in the United States
# and underscores the urgency for ongoing surveillance and effective drug policies.

# Objective 5 Compare death rates between different racial or ethnic groups

# Group by racial/ethnic group in "SEX AND RACE" and calculate average death rate
race_death_avg = df.groupby("SEX AND RACE")["DEATH RATE"].mean().reset_index()

# Sort by death rate (optional)
race_death_avg = race_death_avg.sort_values(by="DEATH RATE", ascending=False)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x="SEX AND RACE", y="DEATH RATE", data=race_death_avg, palette="Spectral")

# Customize plot
plt.xticks(rotation=45, ha='right')
plt.xlabel("Racial/Ethnic Group")
plt.ylabel("Average Death Rate")
plt.title("Average Overdose Death Rate by Racial/Ethnic Group")
plt.tight_layout()

# Show plot
plt.show()

# Conclusion
# There is notable variation in overdose death rates across racial and ethnic groups.
# Groups such as White males and Black males often show higher average rates compared to
# others. This points to potential socioeconomic and healthcare access disparities that
# need targeted intervention and further exploration.

