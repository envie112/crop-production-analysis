# Crop Production Data Analysis

## Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure plots are displayed correctly
sns.set(style="whitegrid")

# Load Dataset
data_path = r"C:\Users\ramya\Desktop\data science\crop production\crop_production.csv"
df = pd.read_csv(data_path)

## Data Cleaning
# Drop duplicates and handle missing values
df.drop_duplicates(inplace=True)
df.dropna(subset=['Production'], inplace=True)

# Replace negative or zero production values with NaN, then drop
invalid_production = df[df['Production'] <= 0]
if not invalid_production.empty:
    print(f"Found {len(invalid_production)} rows with invalid production values.")
df = df[df['Production'] > 0]

# Verify the dataset after cleaning
print("Dataset after cleaning:\n", df.info())
print("\nSample data:\n", df.head())

## Exploratory Data Analysis (EDA)
# Basic statistics and data checks
print("Basic statistics:\n", df.describe())

# Top 10 states by total production
state_totals = df.groupby('State_Name')['Production'].sum().reset_index()
state_totals = state_totals.sort_values('Production', ascending=False)

# Top 10 states
print("\nTop 10 states by production:\n", state_totals.head(10))

# Production by season
season_totals = df.groupby('Season')['Production'].sum().reset_index()
season_totals = season_totals.sort_values('Production', ascending=False)
print("\nProduction by season:\n", season_totals)

## Data Visualization

# 1. Bar chart of top 10 states by production
plt.figure(figsize=(14, 8))
top_10 = state_totals.head(10)

sns.barplot(x='Production', y='State_Name', data=top_10)
plt.title('Top 10 States by Agricultural Production', fontsize=16)
plt.xlabel('Production (in units)', fontsize=12)
plt.ylabel('State_Name', fontsize=12)
plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
plt.tight_layout()
plt.savefig('top_10_states.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Production by season (across all states)
plt.figure(figsize=(12, 6))
sns.barplot(x='Season', y='Production', data=season_totals)
plt.title('Agricultural Production by Season (All States)', fontsize=16)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Production (in units)', fontsize=12)
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('production_by_season.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Heatmap of states and seasons
pivot_data = df.pivot_table(values='Production', index='State_Name', columns='Season', aggfunc='sum', fill_value=0)
pivot_norm = np.log10(pivot_data + 1)

plt.figure(figsize=(16, 14))
sns.heatmap(pivot_norm, cmap='YlGnBu', linewidths=0.5)
plt.title('Heatmap of Agricultural Production by State and Season (Log Scale)', fontsize=16)
plt.xlabel('Season', fontsize=12)
plt.ylabel('State_Name', fontsize=12)
plt.tight_layout()
plt.savefig('state_season_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Pie chart of top 5 states' contribution to total production
total_production = state_totals['Production'].sum()
top_5_states = state_totals.head(5)
other_states = pd.DataFrame({
    'State_Name': ['Others'],
    'Production': [total_production - top_5_states['Production'].sum()]
})
pie_data = pd.concat([top_5_states, other_states])

plt.figure(figsize=(10, 10))
plt.pie(pie_data['Production'], labels=pie_data['State_Name'], autopct='%1.1f%%', startangle=90, shadow=True)
plt.axis('equal')
plt.title('Share of Top 5 States in Total Agricultural Production', fontsize=16)
plt.tight_layout()
plt.savefig('top_5_states_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Seasonal distribution for top 5 states
top_5_states_list = state_totals.head(5)['State_Name'].tolist()
top_5_df = df[df['State_Name'].isin(top_5_states_list)]

plt.figure(figsize=(14, 10))
g = sns.catplot(x='State_Name', y='Production', hue='Season', data=top_5_df, kind='bar', height=6, aspect=2)
plt.title('Seasonal Production Distribution for Top 5 States', fontsize=16)
plt.xlabel('State_Name', fontsize=12)
plt.ylabel('Production (in units)', fontsize=12)
plt.xticks(rotation=45)
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.savefig('top_5_states_seasonal.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Stacked bar chart showing production types by state (for top 10 states)
top_10_states_list = state_totals.head(10)['State_Name'].tolist()
top_10_df = df[df['State_Name'].isin(top_10_states_list)]

pivot_top10 = top_10_df.pivot_table(values='Production', index='State_Name', columns='Season', aggfunc='sum', fill_value=0)
pivot_top10 = pivot_top10.loc[top_10_states_list]

plt.figure(figsize=(16, 10))
pivot_top10.plot(kind='bar', stacked=True)
plt.title('Production by Season for Top 10 States', fontsize=16)
plt.xlabel('State_Name', fontsize=12)
plt.ylabel('Production (in units)', fontsize=12)
plt.xticks(rotation=45)
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('top_10_stacked_season.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Horizontal bar chart showing production by state and season (excluding whole year for better comparison)

seasonal_df = df[df['Season'] != 'Whole Year']
seasonal_pivot = seasonal_df.pivot_table(values='Production', index='State_Name', columns='Season', aggfunc='sum', fill_value=0)

top_states = seasonal_df.groupby('State_Name')['Production'].sum().nlargest(15).index
seasonal_pivot = seasonal_pivot.loc[top_states]
plt.figure(figsize=(14, 12))
ax = seasonal_pivot.plot(kind='barh', figsize=(14, 12), width=0.8)

# Set logarithmic scale for x-axis
ax.set_xscale('log')
plt.title('Seasonal Production by State (Top 15 States)', fontsize=16)
plt.xlabel('Production (log scale)', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add simple value labels for large values
for container in ax.containers:
    # Format labels for the bars
    labels = []
    for v in container.datavalues:
        if v > 1000000:  # Only label values over 1 million
            if v > 1000000000:  # Billions
                labels.append(f'{v/1000000000:.1f}B')
            else:  # Millions
                labels.append(f'{v/1000000:.1f}M')
        else:
            labels.append('')
    
    ax.bar_label(container, labels=labels, padding=5, fontsize=9)

plt.tight_layout()
plt.savefig('seasonal_production_by_state.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Distribution of "Whole Year" vs seasonal production

# Group data by Season and calculate the total production
production_by_season = df.groupby('Season')['Production'].sum().reset_index()
production_by_season = production_by_season.sort_values(by='Production', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Season', y='Production', data=production_by_season, palette='viridis')
plt.yscale('log')
plt.title('Whole Year vs Seasonal Production', fontsize=16)
plt.xlabel('Seasonal Category', fontsize=14)
plt.ylabel('Production (log scale)', fontsize=14)
plt.tight_layout()
plt.savefig('whole_year_vs_seasonal.png', dpi=300, bbox_inches='tight')
plt.close()

# Scatter chart: Production vs State
scatter_data = df.groupby('State_Name')['Production'].sum().reset_index()

plt.figure(figsize=(16, 8))
sns.scatterplot(x='Production', y='State_Name', data=scatter_data, s=100, color='blue', edgecolor='black')
plt.title('Scatter Plot of Agricultural Production by State', fontsize=16)
plt.xlabel('Production (in units)', fontsize=12)
plt.ylabel('State_Name', fontsize=12)
plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('scatter_plot_production_vs_state.png', dpi=300, bbox_inches='tight')
plt.close()

