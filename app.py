import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Ajit E-Commerce Analysis",
    page_icon="📊",
    layout="wide",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff; }
    h1, h2, h3 { color: #ffffff; }
    .insight-box { background-color: rgba(0, 212, 255, 0.1); padding: 15px; border-radius: 5px; border: 1px solid #00d4ff; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_and_merge_data():
    files = {
        'campaigns': 'dim_campaigns.csv',
        'products': 'dim_products.csv',
        'stores': 'dim_stores.csv',
        'events': 'fact_events.csv'
    }
    
    for name, path in files.items():
        if not os.path.exists(path):
            st.error(f"Error: {path} not found. Please upload it.")
            return None
            
    df_campaigns = pd.read_csv(files['campaigns'])
    df_products = pd.read_csv(files['products'])
    df_stores = pd.read_csv(files['stores'])
    df_events = pd.read_csv(files['events'])
    
    # Cleaning
    df_events.drop_duplicates(inplace=True)
    df_events['quantity_sold(before_promo)'].fillna(df_events['quantity_sold(before_promo)'].median(), inplace=True)
    
    # Merging
    df = pd.merge(df_events, df_stores, on='store_id', how='left')
    df = pd.merge(df, df_products, on='product_code', how='left')
    df = pd.merge(df, df_campaigns, on='campaign_id', how='left')
    
    # Calculations
    df['revenue_before_promo'] = df['base_price(before_promo)'] * df['quantity_sold(before_promo)']
    df['revenue_after_promo'] = df['base_price(after_promo)'] * df['quantity_sold(after_promo)']
    df['incremental_revenue'] = df['revenue_after_promo'] - df['revenue_before_promo']
    df['incremental_sold_units'] = df['quantity_sold(after_promo)'] - df['quantity_sold(before_promo)']
    df['ISU_percentage'] = (df['incremental_sold_units'] / df['quantity_sold(before_promo)']) * 100
    
    return df

df = load_and_merge_data()

if df is not None:
    st.title("🛍️ Ajit E-Commerce: Campaign Performance Dashboard")
    st.markdown("---")

    # Sidebar Navigation
    analysis_tabs = [
        "1. Store Distribution",
        "2. Sankranti Category Analysis",
        "3. Price vs Quantity Correlation",
        "4. Pre-Promo Quantity Distribution",
        "5. City-wise ISU% Effectiveness",
        "6. Hyderabad Promo Efficiency",
        "7. Bengaluru Category Performance"
    ]
    choice = st.sidebar.radio("Go to Analysis", analysis_tabs)

    # ---------------------------------------------------------
    # 1. STORE DISTRIBUTION
    # ---------------------------------------------------------
    if choice == "1. Store Distribution":
        st.header("📍 1. Store Distribution Across Cities")
        store_counts = df.groupby('city')['store_id'].nunique().reset_index().sort_values('store_id', ascending=False)
        
        fig1 = px.bar(store_counts, x='city', y='store_id', 
                     text='store_id', color='store_id', 
                     color_continuous_scale='Blues',
                     title="Number of Stores per City")
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **Key Insights:**
            - **Bengaluru** has the highest number of stores (10), followed by Chennai (8) and Hyderabad (7).
            - The distribution shows a concentration in major Tier-1 metro cities, it also indicates the lifestyle of people in these cities and there daily need.
            """)
        with col2:
            ben = store_counts[store_counts['city'] == 'Bengaluru']['store_id'].values[0]
            hyd = store_counts[store_counts['city'] == 'Hyderabad']['store_id'].values[0]
            che = store_counts[store_counts['city'] == 'Chennai']['store_id'].values[0]
            st.metric("Bengaluru vs Others", f"{ben} Stores", f"+{ben-hyd} vs Hyderabad")

    # ---------------------------------------------------------
    # 2. SANKRANTI CATEGORY CONTRIBUTION
    # ---------------------------------------------------------
    elif choice == "2. Sankranti Category Analysis":
        st.header("🌾 2. Sankranti Campaign: Category Contribution")
        sank_df = df[df['campaign_name'] == 'Sankranti']
        cat_sales = sank_df.groupby('category')['quantity_sold(after_promo)'].sum().reset_index()
        
        fig2 = px.pie(cat_sales, values='quantity_sold(after_promo)', names='category', 
                     hole=0.4, title="Category-wise Contribution (Sankranti)",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Analysis:</b> Grocery & Staples and Home Appliances categories typically dominate the sales during Sankranti. 
        This indicates that consumers prioritize essential household items and festival-related staples during this period.
        </div>
        """, unsafe_allow_html=True)

        # Additional Sankranti Insight (from final.py)
        sank_prod_rev = sank_df.groupby('product_name').agg({
            'revenue_before_promo': 'sum',
            'revenue_after_promo': 'sum'
        }).reset_index()
        sank_prod_rev['IR%'] = ((sank_prod_rev['revenue_after_promo'] - sank_prod_rev['revenue_before_promo']) / sank_prod_rev['revenue_before_promo']) * 100
        highest_ir_prod = sank_prod_rev.loc[sank_prod_rev['IR%'].idxmax()]
        
        st.success(f"🚀 **Highest IR% Product**: {highest_ir_prod['product_name']} with **{highest_ir_prod['IR%']:.2f}%** Incremental Revenue")

    # ---------------------------------------------------------
    # 3. PRICE VS QUANTITY CORRELATION
    # ---------------------------------------------------------
    elif choice == "3. Price vs Quantity Correlation":
        st.header("📉 3. Base Price vs Sales Quantity Correlation")
        
        # KDE Plot (Density Heatmap)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.kdeplot(
            data=df,
            x='base_price(after_promo)',
            y='quantity_sold(after_promo)',
            fill=True, cmap='viridis', levels=20, ax=ax3
        )
        ax3.set_title('Density Relationship: Base Price vs Quantity Sold', color='white')
        ax3.set_xlabel('Base Price (After Promo)', color='white')
        ax3.set_ylabel('Quantity Sold (After Promo)', color='white')
        ax3.tick_params(colors='white')
        ax3.grid(True, linestyle='--', alpha=0.3)
        fig3.patch.set_facecolor('#0e1117')
        ax3.set_facecolor('#0e1117')
        
        st.pyplot(fig3)
        
        st.markdown(f"""
        <div class="insight-box">
        <b>Correlation Analysis:</b><br>
        <b>Dense Areas:</b> The brighter or more intensely colored regions on the plot indicate combinations of price and quantity that occur more frequently. We can see a few prominent dense areas:<br><br>
        1. One very dense region appears at lower base prices (after promo), specifically around the 0-200 range, which corresponds to lower quantities sold (after promo), typically below 500 units. This suggests that a significant number of promotions involve products with lower base prices, and these often result in moderate quantities sold.<br>
        2. There's also a denser concentration at higher base prices (after promo), particularly around the 2500-3000 range, which corresponds to moderate quantities sold (after promo), often around 100-300 units. This indicates that even high-priced items under promo can achieve reasonable sales volumes.<br><br>
        <b>Sparse Areas:</b> The darker or less intense areas indicate combinations that are less common.<br><br>
        <b>General Trend:</b> There doesn't appear to be a simple linear relationship. Instead, the plot shows clusters of activity. For instance, very high quantities sold (e.g., above 1500) seem to be associated with a broad range of prices, but perhaps more concentrated towards the lower end, while very high base prices tend to be associated with lower to moderate quantities.<br><br>
        In essence, the plot helps to identify the most common price-quantity pairings after promotions, revealing that both very low-priced items and very high-priced items (within specific ranges) contribute significantly to sales quantities, but with different distribution patterns.
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4. PRE-PROMO QUANTITY DISTRIBUTION
    # ---------------------------------------------------------
    elif choice == "4. Pre-Promo Quantity Distribution":
        st.header("📊 4. Distribution of Quantity Sold (Before Promotion)")
        
        cats = df['category'].unique()
        selected_cat = st.selectbox("Select Category to View Distribution", cats)
        
        filtered_cat = df[df['category'] == selected_cat]
        fig4 = px.histogram(filtered_cat, x='quantity_sold(before_promo)', 
                           nbins=30, marginal="box", color_discrete_sequence=['#FF4B4B'],
                           title=f"Distribution for {selected_cat}")
        st.plotly_chart(fig4, use_container_width=True)
        
        st.info("""
        Establishing a Baseline: The 'quantity_sold(before_promo)' data is your benchmark. It tells you the natural sales volume for each category without any promotional push. This baseline is essential for accurately measuring the incremental impact of any future promotion. If a promotion sells 500 units, but the product normally sells 400, the true promotional uplift is only 100.

Tailoring Promotion Types:

High Baseline Sales (e.g., Grocery & Staples): For categories like 'Grocery & Staples' with high average 'before promo' sales, promotions might aim to increase market share, encourage larger basket sizes, or drive repeat purchases. Strategies like 'Buy One Get One Free' (BOGOF) or percentage-off discounts could be highly effective in moving even greater volumes.
Low Baseline Sales (e.g., Home Appliances, Home Care, Personal Care): For categories with lower natural sales, promotions might need to be more aggressive to encourage trial or first-time purchases. Consider:
Deeper Discounts: To overcome higher price points.
Bundling: Combining lower-selling items with complementary products to increase perceived value.
Cashback Offers: Especially for higher-value items like 'Home Appliances', cashback can be more appealing than a direct price reduction.
Awareness Campaigns: If sales are low, it might also indicate a lack of product awareness, so promotions could be coupled with marketing efforts.
Optimizing Promotional Depth and Frequency:

Consistent Sales (low standard deviation): Categories with relatively stable baseline sales might benefit from smaller, more frequent promotions to maintain customer engagement without significantly eroding margins.
Variable Sales (high standard deviation): Categories with more fluctuating sales might require more impactful, but potentially less frequent, promotions to create a significant spike in demand when needed.
Resource Allocation: Understanding the natural demand allows for more strategic allocation of promotional budgets. Categories with high inherent demand might require less heavy investment in promotions to achieve sales targets, freeing up resources for categories that genuinely need a boost.

Inventory Management: Baseline sales data directly informs inventory decisions. Consistently high-selling categories require robust stocking, while lower-selling ones might need more cautious inventory management to avoid overstocking.

By understanding these underlying sales patterns, businesses can move beyond generic promotions to highly targeted strategies that maximize return on investment for each product category.

        """)

    # ---------------------------------------------------------
    # 5. CITY-WISE ISU% EFFECTIVENESS
    # ---------------------------------------------------------
    elif choice == "5. City-wise ISU% Effectiveness":
        st.header("📈 5. Incremental Sold Units Percentage (ISU%) by City")
        
        city_isu = df.groupby('city').agg({
            'quantity_sold(before_promo)': 'sum',
            'quantity_sold(after_promo)': 'sum'
        }).reset_index()
        city_isu['ISU%'] = ((city_isu['quantity_sold(after_promo)'] - city_isu['quantity_sold(before_promo)']) / city_isu['quantity_sold(before_promo)']) * 100
        city_isu = city_isu.sort_values('ISU%', ascending=False)
        
        fig5 = px.line(city_isu, x='city', y='ISU%', markers=True, 
                      title="Promotion Effectiveness (ISU%) Comparison",
                      labels={'ISU%': 'Incremental Sold Units %'})
        fig5.update_traces(line_color='#00d4ff', line_width=3)
        st.plotly_chart(fig5, use_container_width=True)
        
        top_city = city_isu.iloc[0]
        low_city = city_isu.iloc[-1]
        
        st.success(f"🏆 **Highest ISU%**: {top_city['city']} ({top_city['ISU%']:.2f}%)")
        st.warning(f"📉 **Smallest Change**: {low_city['city']} ({low_city['ISU%']:.2f}%)")

        st.markdown("""
        ### City-wise Effectiveness Analysis:
        * **Overall Positive Impact:** All cities show a positive incremental sold unit percentage, indicating that promotions are generally effective in driving higher unit sales across all locations.
        * **Highest Effectiveness:** Madurai stands out with the highest incremental sold unit percentage, suggesting that promotions have been most impactful there.
        * **Lowest Effectiveness:** Visakhapatnam has the lowest incremental sold unit percentage. While still showing an increase, promotions there appear to be relatively less effective.
        * **Consistent Performance:** Most cities exhibit a similar range of incremental sales percentages (between 100% and 115%), suggesting a fairly consistent uplift in sales across a majority of locations.
        """)

    # ---------------------------------------------------------
    # 6. HYDERABAD PROMO EFFICIENCY
    # ---------------------------------------------------------
    elif choice == "6. Hyderabad Promo Efficiency":
        st.header("⚡ 6. Promotion Efficiency: Hyderabad Deep Dive")
        hyd_df = df[df['city'] == 'Hyderabad'].copy()
        
        # Calculate percentages by promo type
        promo_metrics = hyd_df.groupby('promo_type').agg({
            'revenue_before_promo': 'sum',
            'revenue_after_promo': 'sum',
            'quantity_sold(before_promo)': 'sum',
            'quantity_sold(after_promo)': 'sum'
        }).reset_index()
        
        promo_metrics['incremental_revenue_change_%'] = ((promo_metrics['revenue_after_promo'] - promo_metrics['revenue_before_promo']) / promo_metrics['revenue_before_promo']) * 100
        promo_metrics['incremental_quantity_change_%'] = ((promo_metrics['quantity_sold(after_promo)'] - promo_metrics['quantity_sold(before_promo)']) / promo_metrics['quantity_sold(before_promo)'] ) * 100
        
        # Seaborn Plot with Regression Line
        fig6, ax6 = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=promo_metrics,
            x='incremental_quantity_change_%',
            y='incremental_revenue_change_%',
            hue='promo_type',
            s=200, palette='viridis', ax=ax6
        )
        
        sns.regplot(
            data=promo_metrics,
            x='incremental_quantity_change_%',
            y='incremental_revenue_change_%',
            scatter=False, color='red',
            line_kws={'linestyle': '--', 'linewidth': 2},
            ax=ax6, label='Regression Line'
        )
        
        # Add labels to points
        for i, row in promo_metrics.iterrows():
            ax6.text(row['incremental_quantity_change_%'] + 1, row['incremental_revenue_change_%'] + 1, row['promo_type'], color='white', fontsize=9)
            
        ax6.set_title('Incremental Quantity vs Revenue Change (Hyderabad)', color='white')
        ax6.set_xlabel('Incremental Quantity Change (%)', color='white')
        ax6.set_ylabel('Incremental Revenue Change (%)', color='white')
        ax6.tick_params(colors='white')
        ax6.grid(True, linestyle='--', alpha=0.3)
        ax6.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax6.axvline(0, color='grey', linestyle='--', linewidth=0.8)
        ax6.legend()
        fig6.patch.set_facecolor('#0e1117')
        ax6.set_facecolor('#0e1117')
        
        st.pyplot(fig6)
        
        st.markdown("""
        General Trend: The regression line shows a clear positive slope. This indicates that, generally, as the incremental quantity sold increases, the incremental revenue also tends to increase. This is an expected positive correlation.

Outliers/Strong Performers:

The 500 Cashback and BOGOF promotions are located significantly above the regression line. This means they are generating more incremental revenue for a given incremental quantity increase than what the general trend would predict. They are highly efficient and effective.
Underperformers:

The 25% OFF promotion is far below the regression line, positioned in the negative quadrant for both quantity and revenue. It's the least effective, significantly underperforming the general trend.
The 33% OFF and 50% OFF promotions, while showing positive incremental quantity changes, fall below or very close to the regression line, especially in terms of revenue. This confirms that these promotions, despite driving some quantity, are less efficient in generating incremental revenue compared to the other top-performing promo types, as they result in significant revenue reduction relative to the sales volume.
Strategic Implications:

Focus on High-Leverage Promotions: Promotions like '500 Cashback' and 'BOGOF' are high-leverage and should be prioritized in Hyderabad, as they deliver superior incremental revenue for the additional units sold.
Re-evaluate Discount-Based Promotions: The percentage-off discounts, particularly '25% OFF', are less effective. It's crucial to either adjust their mechanics (e.g., higher price points, different product categories) or reconsider their use if the goal is incremental revenue growth in Hyderabad.
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 7. BENGALURU CATEGORY PERFORMANCE
    # ---------------------------------------------------------
    elif choice == "7. Bengaluru Category Performance":
        st.header("🏢 7. Revenue Performance by Category: Bengaluru")
        ben_df = df[df['city'] == 'Bengaluru'].copy()
        
        revenue_by_category_bengaluru = ben_df.groupby('category').agg({
            'revenue_before_promo': 'sum',
            'revenue_after_promo': 'sum'
        }).reset_index()
        
        revenue_by_category_bengaluru['incremental_revenue_change_%'] = (
            (revenue_by_category_bengaluru['revenue_after_promo'] - revenue_by_category_bengaluru['revenue_before_promo']) /
            revenue_by_category_bengaluru['revenue_before_promo']
        ) * 100
        
        # Seaborn Bar Plot
        fig7, ax7 = plt.subplots(figsize=(12, 7))
        sns.barplot(
            x='category',
            y='incremental_revenue_change_%',
            data=revenue_by_category_bengaluru,
            hue='category',
            palette='viridis',
            legend=False,
            ax=ax7
        )
        ax7.set_title('Incremental Revenue Change by Product Category in Bengaluru', color='white')
        ax7.set_xlabel('Product Category', color='white')
        ax7.set_ylabel('Incremental Revenue Change (%)', color='white')
        ax7.tick_params(colors='white', rotation=45)
        ax7.grid(axis='y', linestyle='--', alpha=0.3)
        fig7.patch.set_facecolor('#0e1117')
        ax7.set_facecolor('#0e1117')
        plt.tight_layout()
        
        st.pyplot(fig7)
        
        st.markdown("""
        Combo1 (141.65%): This category shows an exceptionally high incremental revenue percentage, indicating that promotions are highly effective and drive significant revenue growth for combo products.

Home Appliances (88.35%): Another strong performer, home appliances benefit greatly from promotions, yielding substantial incremental revenue.

Home Care (51.11%): Promotions in the home care category also lead to good incremental revenue, though less dramatically than Combo1 or Home Appliances.

Grocery & Staples (13.04%): While still positive, the incremental revenue for grocery and staples is more modest. Promotions here contribute to revenue growth but with less impact relative to their baseline.

Personal Care (-32.39%): This category stands out as the only one with a negative incremental revenue percentage. This suggests that promotions in the personal care segment in Bengaluru are either poorly designed, attract customers primarily looking for discounts without increasing overall spend, or are leading to significant margin erosion, ultimately reducing revenue.

Overall Impact on Bengaluru Revenue:
Overall, promotions in Bengaluru appear to have a positive impact on total revenue, driven strongly by the success of promotions in the Combo1 and Home Appliances categories. However, the negative performance in Personal Care indicates a need for re-evaluation of promotional strategies for this specific category to prevent revenue losses. The positive, albeit smaller, increases in Home Care and Grocery & Staples further contribute to the city's overall positive promotional outcomes.
        """)

else:
    st.warning("Please ensure your CSV files are in the directory.")
