"""Generate a comprehensive HTML report for M.F. Husain auction analysis."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import re

# Set dark template for all figures
pio.templates.default = "plotly_dark"

# ── Load & filter data ──────────────────────────────────────────────────────
master = pd.read_csv('data/processed/master.csv')
husain = master[master['artist_name'] == 'MAQBOOL FIDA HUSAIN'].copy()
husain['auction_date'] = pd.to_datetime(husain['auction_date'])
sold = husain[husain['is_sold'] == True].copy()
unsold = husain[husain['is_sold'] == False].copy()

# ── Clean medium ────────────────────────────────────────────────────────────
MEDIUM_MAP = {
    'oil_on_canvas': 'Oil on Canvas', 'acrylic_on_canvas': 'Acrylic on Canvas',
    'acrylic_on_board': 'Acrylic on Board', 'oil_on_board': 'Oil on Board',
    'watercolor': 'Watercolor', 'watercolor_on_paper': 'Watercolor',
    'gouache': 'Works on Paper', 'pastel_on_paper': 'Works on Paper',
    'ink_on_paper': 'Works on Paper', 'ink_wash_on_paper': 'Works on Paper',
    'charcoal_on_paper': 'Works on Paper', 'pencil_on_paper': 'Works on Paper',
    'acrylic_on_paper': 'Works on Paper', 'oil_on_paper': 'Works on Paper',
    'oil_other': 'Oil on Canvas', 'mixed_media': 'Mixed Media',
    'wood_sculpture': 'Sculpture', 'print': 'Print', 'lithograph': 'Print',
    'other': 'Other',
}
husain['medium_clean'] = husain['medium_category'].map(MEDIUM_MAP).fillna('Other')
sold['medium_clean'] = sold['medium_category'].map(MEDIUM_MAP).fillna('Other')

# ── Theme extraction from titles ────────────────────────────────────────────
THEME_PATTERNS = {
    'Horses': r'(?i)\b(horse|horses|equus|zuljinah|rearing horse)\b',
    'Women & Figures': r'(?i)\b(woman|women|lady|ladies|girl|nude|nudes|seated lady|seated woman|three graces|bathers|mother and daughter|tribal women|portrait of indrani|ragini|meera|gajagamini|valentina|fertility)\b',
    'Religion & Mythology': r'(?i)\b(ganesha|hanuman|durga|shiva|parvati|ramayana|krishna|buddha|buddhism|madonna|naga|kailash|ardhnareshwar|vishnu)\b',
    'Music & Dance': r'(?i)\b(musician|musicians|drummer|sitar|mridang|tabalchi|puppet dancers|street musicians|ustadjan)\b',
    'Rural & Village Life': r'(?i)\b(village|farmer|kisan|bail gadi|basket weaver|rural|rajasthan|kashmir|kerala|gujarat|banaras|tribal|rajasthani)\b',
    'Mother Teresa': r'(?i)\bmother teresa\b',
    'Portraits & Self-Portraits': r'(?i)\b(portrait|self.portrait|autobiography)\b',
    'Animals (non-horse)': r'(?i)\b(elephant|leopard|bull|bulls|bestiary)\b',
    'Film & Pop Culture': r'(?i)\b(devdas|cinema|film|madhuri|mohini)\b',
    'Abstract & Other': r'(?i)\b(abstract|fantasy|design|concorde|lightning|wilderness|question|hostage|pull|wind|yatra|sindoor)\b',
}

def classify_theme(title):
    if pd.isna(title):
        return 'Unknown'
    themes = []
    for theme, pattern in THEME_PATTERNS.items():
        if re.search(pattern, str(title)):
            themes.append(theme)
    return themes[0] if themes else 'Other'

husain['theme'] = husain['title'].apply(classify_theme)
sold['theme'] = sold['title'].apply(classify_theme)

# ── Size buckets ────────────────────────────────────────────────────────────
def size_bucket(area):
    if pd.isna(area):
        return 'Unknown'
    if area < 1000:
        return 'Small (<1000 cm²)'
    elif area < 3000:
        return 'Medium (1000-3000 cm²)'
    elif area < 6000:
        return 'Large (3000-6000 cm²)'
    else:
        return 'Very Large (>6000 cm²)'

husain['size_bucket'] = husain['surface_area_cm2'].apply(size_bucket)
sold['size_bucket'] = sold['surface_area_cm2'].apply(size_bucket)

# ── Decade buckets ──────────────────────────────────────────────────────────
def decade_bucket(year):
    if pd.isna(year):
        return 'Unknown'
    return f"{int(year // 10 * 10)}s"

husain['decade'] = husain['year_created'].apply(decade_bucket)
sold['decade'] = sold['year_created'].apply(decade_bucket)

# ── Estimate helpers ────────────────────────────────────────────────────────
husain['estimate_avg'] = husain[['estimate_low_usd', 'estimate_high_usd']].mean(axis=1)
sold['estimate_avg'] = sold[['estimate_low_usd', 'estimate_high_usd']].mean(axis=1)
sold['estimate_deviation'] = abs((sold['hammer_price_usd'] - sold['estimate_avg']) / sold['hammer_price_usd'])
sold['over_estimate'] = (sold['hammer_price_usd'] > sold['estimate_high_usd']).astype(int)
sold['under_estimate'] = (sold['hammer_price_usd'] < sold['estimate_low_usd']).astype(int)
sold['within_estimate'] = ((sold['hammer_price_usd'] >= sold['estimate_low_usd']) &
                           (sold['hammer_price_usd'] <= sold['estimate_high_usd'])).astype(int)

# ═══════════════════════════════════════════════════════════════════════════
#  BUILD CHARTS
# ═══════════════════════════════════════════════════════════════════════════
charts = {}
COLORS = px.colors.qualitative.Set2

_first_chart = True
def chart_html(fig):
    """Convert figure to HTML fragment, embedding plotly.js only in the first chart."""
    global _first_chart
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    html = fig.to_html(full_html=False, include_plotlyjs=_first_chart)
    _first_chart = False
    return html

# ── 1. Price Evolution Year over Year ───────────────────────────────────────
yearly = sold.groupby('auction_year')['hammer_price_usd'].agg(
    ['mean', 'median', 'min', 'max', 'count', 'std']
).reset_index()
yearly.columns = ['Year', 'Mean', 'Median', 'Low', 'High', 'Count', 'Std']

fig = go.Figure()
fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['High'], mode='lines+markers',
                         name='High', line=dict(color='#e74c3c', width=2)))
fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Mean'], mode='lines+markers',
                         name='Mean', line=dict(color='#3498db', width=3)))
fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Median'], mode='lines+markers',
                         name='Median', line=dict(color='#2ecc71', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Low'], mode='lines+markers',
                         name='Low', line=dict(color='#f39c12', width=2)))
fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['Count'], name='# Sold',
                     yaxis='y2', opacity=0.2, marker_color='#95a5a6'))
fig.update_layout(
    title='Price Evolution Year over Year',
    xaxis_title='Auction Year', yaxis_title='Price (USD)',
    yaxis2=dict(title='# Lots Sold', overlaying='y', side='right'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    height=500,
)
charts['price_evolution'] = chart_html(fig)

# ── 2. Total Sales & Sell-Through by Year ───────────────────────────────────
lots_yr = husain.groupby('auction_year').size().reset_index(name='total')
sold_yr = sold.groupby('auction_year').size().reset_index(name='sold')
merged_yr = lots_yr.merge(sold_yr, on='auction_year', how='left').fillna(0)
merged_yr['sell_through'] = merged_yr['sold'] / merged_yr['total'] * 100
merged_yr['total_sales'] = sold.groupby('auction_year')['hammer_price_usd'].sum().values if len(sold_yr) == len(lots_yr) else 0

# recalc total_sales properly
total_sales_yr = sold.groupby('auction_year')['hammer_price_usd'].sum().reset_index(name='total_sales')
merged_yr = merged_yr.drop(columns='total_sales', errors='ignore').merge(total_sales_yr, on='auction_year', how='left').fillna(0)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=merged_yr['auction_year'], y=merged_yr['total_sales'],
                     name='Total Sales (USD)', marker_color='#3498db'), secondary_y=False)
fig.add_trace(go.Scatter(x=merged_yr['auction_year'], y=merged_yr['sell_through'],
                         name='Sell-Through %', mode='lines+markers',
                         line=dict(color='#e74c3c', width=3)), secondary_y=True)
fig.update_layout(title='Total Sales & Sell-Through Rate by Year',
                  xaxis_title='Year', height=450,
                  legend=dict(orientation='h', yanchor='bottom', y=1.02))
fig.update_yaxes(title_text='Total Sales (USD)', secondary_y=False)
fig.update_yaxes(title_text='Sell-Through %', secondary_y=True, range=[0, 100])
charts['sales_sellthrough'] = chart_html(fig)

# ── 3. Lots per Year: Total vs Sold ────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Bar(x=merged_yr['auction_year'], y=merged_yr['total'],
                     name='Total Lots', marker_color='#bdc3c7'))
fig.add_trace(go.Bar(x=merged_yr['auction_year'], y=merged_yr['sold'],
                     name='Sold', marker_color='#2c3e50'))
fig.update_layout(title='Lots per Year: Total vs. Sold', barmode='group',
                  xaxis_title='Year', yaxis_title='# Lots', height=400)
charts['lots_per_year'] = chart_html(fig)

# ── 4. Medium Distribution (all lots) ──────────────────────────────────────
med_counts = husain['medium_clean'].value_counts()
fig = go.Figure(go.Pie(labels=med_counts.index, values=med_counts.values,
                       textinfo='label+percent', hole=0.3))
fig.update_layout(title='Distribution of Mediums (All Lots)', height=450)
charts['medium_pie'] = chart_html(fig)

# ── 5. Price by Medium (sold) ──────────────────────────────────────────────
med_price = sold.groupby('medium_clean')['hammer_price_usd'].agg(['mean', 'median', 'count']).reset_index()
med_price.columns = ['Medium', 'Mean', 'Median', 'Count']
med_price = med_price.sort_values('Mean', ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(y=med_price['Medium'], x=med_price['Mean'], orientation='h',
                     name='Mean Price', marker_color='#3498db',
                     text=[f"${v:,.0f} (n={c})" for v, c in zip(med_price['Mean'], med_price['Count'])],
                     textposition='outside'))
fig.add_trace(go.Bar(y=med_price['Medium'], x=med_price['Median'], orientation='h',
                     name='Median Price', marker_color='#2ecc71'))
fig.update_layout(title='Price by Medium (Sold Lots)', barmode='group',
                  xaxis_title='Price (USD)', height=400, margin=dict(l=120))
charts['medium_price'] = chart_html(fig)

# ── 6. Price by Theme (sold) ───────────────────────────────────────────────
theme_price = sold.groupby('theme')['hammer_price_usd'].agg(['mean', 'median', 'count']).reset_index()
theme_price.columns = ['Theme', 'Mean', 'Median', 'Count']
theme_price = theme_price.sort_values('Mean', ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(y=theme_price['Theme'], x=theme_price['Mean'], orientation='h',
                     name='Mean Price', marker_color='#9b59b6',
                     text=[f"${v:,.0f} (n={c})" for v, c in zip(theme_price['Mean'], theme_price['Count'])],
                     textposition='outside'))
fig.add_trace(go.Bar(y=theme_price['Theme'], x=theme_price['Median'], orientation='h',
                     name='Median Price', marker_color='#e67e22'))
fig.update_layout(title='Price by Theme/Subject (Sold Lots)', barmode='group',
                  xaxis_title='Price (USD)', height=500, margin=dict(l=180))
charts['theme_price'] = chart_html(fig)

# ── 7. Theme distribution (all lots) ───────────────────────────────────────
theme_counts = husain['theme'].value_counts()
fig = go.Figure(go.Pie(labels=theme_counts.index, values=theme_counts.values,
                       textinfo='label+percent', hole=0.3))
fig.update_layout(title='Theme/Subject Distribution (All Lots)', height=450)
charts['theme_pie'] = chart_html(fig)

# ── 8. Price by Size Bucket ────────────────────────────────────────────────
size_order = ['Small (<1000 cm²)', 'Medium (1000-3000 cm²)', 'Large (3000-6000 cm²)', 'Very Large (>6000 cm²)']
size_price = sold.groupby('size_bucket')['hammer_price_usd'].agg(['mean', 'median', 'count']).reset_index()
size_price.columns = ['Size', 'Mean', 'Median', 'Count']
size_price['Size'] = pd.Categorical(size_price['Size'], categories=size_order, ordered=True)
size_price = size_price.sort_values('Size')

fig = go.Figure()
fig.add_trace(go.Bar(x=size_price['Size'], y=size_price['Mean'], name='Mean Price',
                     marker_color='#1abc9c',
                     text=[f"${v:,.0f}<br>n={c}" for v, c in zip(size_price['Mean'], size_price['Count'])],
                     textposition='outside'))
fig.add_trace(go.Bar(x=size_price['Size'], y=size_price['Median'], name='Median Price',
                     marker_color='#16a085'))
fig.update_layout(title='Price by Size of Work (Sold Lots)', barmode='group',
                  xaxis_title='Size Category', yaxis_title='Price (USD)', height=450)
charts['size_price'] = chart_html(fig)

# ── 9. Price by Decade Created ──────────────────────────────────────────────
decade_price = sold[sold['decade'] != 'Unknown'].groupby('decade')['hammer_price_usd'].agg(
    ['mean', 'median', 'count']).reset_index()
decade_price.columns = ['Decade', 'Mean', 'Median', 'Count']
decade_price = decade_price.sort_values('Decade')

fig = go.Figure()
fig.add_trace(go.Bar(x=decade_price['Decade'], y=decade_price['Mean'], name='Mean Price',
                     marker_color='#e74c3c',
                     text=[f"${v:,.0f}<br>n={c}" for v, c in zip(decade_price['Mean'], decade_price['Count'])],
                     textposition='outside'))
fig.add_trace(go.Bar(x=decade_price['Decade'], y=decade_price['Median'], name='Median Price',
                     marker_color='#c0392b'))
fig.update_layout(title='Price by Decade of Creation (Sold Lots)', barmode='group',
                  xaxis_title='Decade Created', yaxis_title='Price (USD)', height=450)
charts['decade_price'] = chart_html(fig)

# ── 10. Scatter: Price vs Surface Area ──────────────────────────────────────
fig = px.scatter(sold.dropna(subset=['surface_area_cm2']),
                 x='surface_area_cm2', y='hammer_price_usd',
                 color='medium_clean', hover_data=['title', 'auction_year'],
                 trendline='ols', title='Price vs. Surface Area by Medium')
fig.update_layout(xaxis_title='Surface Area (cm²)', yaxis_title='Hammer Price (USD)', height=500)
charts['scatter_size_price'] = chart_html(fig)

# ── 11. Scatter: Year Created vs Price ──────────────────────────────────────
fig = px.scatter(sold.dropna(subset=['year_created']),
                 x='year_created', y='hammer_price_usd',
                 color='medium_clean', hover_data=['title'],
                 trendline='ols', title='Year Created vs. Price')
fig.update_layout(xaxis_title='Year Created', yaxis_title='Hammer Price (USD)', height=500)
charts['scatter_year_price'] = chart_html(fig)

# ── 12. Estimate Accuracy ──────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(x=sold['estimate_avg'], y=sold['hammer_price_usd'],
                         mode='markers', marker=dict(size=10, opacity=0.6, color='#3498db'),
                         text=sold['title'], name='Lots'))
max_val = max(sold['estimate_avg'].max(), sold['hammer_price_usd'].max()) * 1.1
fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                         line=dict(dash='dash', color='red'), name='Perfect Estimate'))
fig.update_layout(title="Estimate Accuracy: Estimate vs. Hammer Price",
                  xaxis_title='Estimate Average (USD)', yaxis_title='Hammer Price (USD)', height=500)
charts['estimate_accuracy'] = chart_html(fig)

# ── 13. Estimate Outcome Breakdown ─────────────────────────────────────────
outcome_counts = pd.Series({
    'Over Estimate': sold['over_estimate'].sum(),
    'Within Estimate': sold['within_estimate'].sum(),
    'Under Estimate': sold['under_estimate'].sum(),
})
fig = go.Figure(go.Pie(labels=outcome_counts.index, values=outcome_counts.values,
                       marker_colors=['#2ecc71', '#3498db', '#e74c3c'],
                       textinfo='label+percent+value', hole=0.4))
fig.update_layout(title='How Often Do Lots Beat/Meet/Miss Estimates?', height=400)
charts['estimate_outcome'] = chart_html(fig)

# ── 14. Sell-Through by Medium ──────────────────────────────────────────────
med_total = husain.groupby('medium_clean').size().reset_index(name='total')
med_sold = sold.groupby('medium_clean').size().reset_index(name='sold')
med_st = med_total.merge(med_sold, on='medium_clean', how='left').fillna(0)
med_st['sell_through'] = med_st['sold'] / med_st['total'] * 100
med_st = med_st.sort_values('sell_through', ascending=True)

fig = go.Figure(go.Bar(y=med_st['medium_clean'], x=med_st['sell_through'], orientation='h',
                       marker_color='#27ae60',
                       text=[f"{v:.0f}% ({int(s)}/{int(t)})" for v, s, t in
                             zip(med_st['sell_through'], med_st['sold'], med_st['total'])],
                       textposition='outside'))
fig.update_layout(title='Sell-Through Rate by Medium', xaxis_title='Sell-Through %',
                  height=400, margin=dict(l=120), xaxis_range=[0, 110])
charts['st_medium'] = chart_html(fig)

# ── 15. Sell-Through by Theme ───────────────────────────────────────────────
th_total = husain.groupby('theme').size().reset_index(name='total')
th_sold = sold.groupby('theme').size().reset_index(name='sold')
th_st = th_total.merge(th_sold, on='theme', how='left').fillna(0)
th_st['sell_through'] = th_st['sold'] / th_st['total'] * 100
th_st = th_st.sort_values('sell_through', ascending=True)

fig = go.Figure(go.Bar(y=th_st['theme'], x=th_st['sell_through'], orientation='h',
                       marker_color='#8e44ad',
                       text=[f"{v:.0f}% ({int(s)}/{int(t)})" for v, s, t in
                             zip(th_st['sell_through'], th_st['sold'], th_st['total'])],
                       textposition='outside'))
fig.update_layout(title='Sell-Through Rate by Theme', xaxis_title='Sell-Through %',
                  height=500, margin=dict(l=180), xaxis_range=[0, 110])
charts['st_theme'] = chart_html(fig)

# ── 16. Sale Type (Live vs Online) ─────────────────────────────────────────
sale_counts = husain['sale_type'].value_counts()
fig = go.Figure(go.Pie(labels=sale_counts.index, values=sale_counts.values,
                       textinfo='label+percent+value', hole=0.3,
                       marker_colors=['#3498db', '#e67e22']))
fig.update_layout(title='Auction Type Distribution', height=400)
charts['sale_type'] = chart_html(fig)

# ── 17. Top 15 Most Expensive Works ────────────────────────────────────────
top15 = sold.nlargest(15, 'hammer_price_usd')[
    ['title', 'medium_clean', 'hammer_price_usd', 'surface_area_cm2',
     'year_created', 'auction_year', 'estimate_avg']].copy()
top15['hammer_price_usd'] = top15['hammer_price_usd'].apply(lambda x: f"${x:,.0f}")
top15['estimate_avg'] = top15['estimate_avg'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else 'N/A')
top15['surface_area_cm2'] = top15['surface_area_cm2'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
top15['year_created'] = top15['year_created'].apply(lambda x: f"{int(x)}" if pd.notna(x) else 'N/A')
top15.columns = ['Title', 'Medium', 'Hammer Price', 'Area (cm²)', 'Year Created', 'Year Sold', 'Estimate Avg']

# ── 18. Signed vs Unsigned ─────────────────────────────────────────────────
signed_data = []
for label, mask in [('Signed', sold['is_signed'] == True), ('Unsigned', sold['is_signed'] == False)]:
    subset = sold[mask]['hammer_price_usd']
    if len(subset) > 0:
        signed_data.append({'Status': label, 'Mean': subset.mean(), 'Median': subset.median(), 'Count': len(subset)})
signed_df = pd.DataFrame(signed_data)

fig = go.Figure()
fig.add_trace(go.Bar(x=signed_df['Status'], y=signed_df['Mean'], name='Mean',
                     marker_color='#3498db',
                     text=[f"${v:,.0f} (n={c})" for v, c in zip(signed_df['Mean'], signed_df['Count'])],
                     textposition='outside'))
fig.add_trace(go.Bar(x=signed_df['Status'], y=signed_df['Median'], name='Median',
                     marker_color='#2ecc71'))
fig.update_layout(title='Signed vs. Unsigned Price Comparison', barmode='group',
                  yaxis_title='Price (USD)', height=400)
charts['signed'] = chart_html(fig)

# ── 19. Heatmap: Medium x Decade avg price ─────────────────────────────────
sold_known = sold[(sold['decade'] != 'Unknown')].copy()
pivot = sold_known.pivot_table(values='hammer_price_usd', index='medium_clean',
                                columns='decade', aggfunc='mean')
pivot_count = sold_known.pivot_table(values='hammer_price_usd', index='medium_clean',
                                      columns='decade', aggfunc='count').fillna(0)

# Format text
text_vals = []
for i in range(len(pivot)):
    row = []
    for j in range(len(pivot.columns)):
        val = pivot.iloc[i, j]
        cnt = pivot_count.iloc[i, j]
        if pd.notna(val):
            row.append(f"${val:,.0f}<br>n={int(cnt)}")
        else:
            row.append("")
    text_vals.append(row)

fig = go.Figure(go.Heatmap(
    z=pivot.values, x=pivot.columns, y=pivot.index,
    text=text_vals, texttemplate="%{text}", colorscale='YlOrRd',
    colorbar_title='Avg Price (USD)',
))
fig.update_layout(title='Average Price Heatmap: Medium x Decade',
                  xaxis_title='Decade Created', yaxis_title='Medium', height=450)
charts['heatmap'] = chart_html(fig)

# ── 20. Box plot: Price distribution by medium ──────────────────────────────
fig = px.box(sold, x='medium_clean', y='hammer_price_usd', color='medium_clean',
             title='Price Distribution by Medium', points='all',
             hover_data=['title', 'auction_year'])
fig.update_layout(xaxis_title='Medium', yaxis_title='Price (USD)', height=500, showlegend=False)
charts['box_medium'] = chart_html(fig)

# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY STATS
# ═══════════════════════════════════════════════════════════════════════════
stats = {
    'total_lots': len(husain),
    'total_sold': len(sold),
    'total_unsold': len(unsold),
    'sell_through': f"{len(sold)/len(husain)*100:.1f}%",
    'avg_price': f"${sold['hammer_price_usd'].mean():,.0f}",
    'median_price': f"${sold['hammer_price_usd'].median():,.0f}",
    'max_price': f"${sold['hammer_price_usd'].max():,.0f}",
    'min_price': f"${sold['hammer_price_usd'].min():,.0f}",
    'total_revenue': f"${sold['hammer_price_usd'].sum():,.0f}",
    'avg_deviation': f"{sold['estimate_deviation'].mean()*100:.1f}%",
    'pct_over': f"{sold['over_estimate'].mean()*100:.0f}%",
    'pct_within': f"{sold['within_estimate'].mean()*100:.0f}%",
    'pct_under': f"{sold['under_estimate'].mean()*100:.0f}%",
    'years_range': f"{int(husain['auction_year'].min())}-{int(husain['auction_year'].max())}",
    'creation_range': f"{int(husain['year_created'].dropna().min())}-{int(husain['year_created'].dropna().max())}",
    'num_mediums': len(husain['medium_clean'].unique()),
    'num_themes': len(husain['theme'].unique()),
}

# best/worst performers
best_medium = med_price.sort_values('Mean', ascending=False).iloc[0]
best_theme = theme_price.sort_values('Mean', ascending=False).iloc[0]
best_decade = decade_price.sort_values('Mean', ascending=False).iloc[0]
best_size = size_price.sort_values('Mean', ascending=False).iloc[0]

# ═══════════════════════════════════════════════════════════════════════════
#  BUILD HTML
# ═══════════════════════════════════════════════════════════════════════════
top15_html = top15.to_html(index=False, classes='data-table', border=0)

# Price evolution table
yearly_table = yearly.copy()
yearly_table['Mean'] = yearly_table['Mean'].apply(lambda x: f"${x:,.0f}")
yearly_table['Median'] = yearly_table['Median'].apply(lambda x: f"${x:,.0f}")
yearly_table['Low'] = yearly_table['Low'].apply(lambda x: f"${x:,.0f}")
yearly_table['High'] = yearly_table['High'].apply(lambda x: f"${x:,.0f}")
yearly_table['Std'] = yearly_table['Std'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else 'N/A')
yearly_table.columns = ['Year', 'Mean', 'Median', 'Low', 'High', '# Sold', 'Std Dev']
yearly_table_html = yearly_table.to_html(index=False, classes='data-table', border=0)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>M.F. Husain - Auction Market Analysis</title>
<style>
  :root {{
    --bg: #0f0f0f; --card: #1a1a2e; --accent: #e94560; --accent2: #0f3460;
    --text: #eee; --text2: #aaa; --border: #2a2a4a;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  header {{ text-align: center; padding: 60px 20px 40px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); border-bottom: 3px solid var(--accent); }}
  header h1 {{ font-size: 2.8em; font-weight: 300; letter-spacing: 2px; }}
  header h1 span {{ color: var(--accent); font-weight: 600; }}
  header p {{ color: var(--text2); font-size: 1.1em; margin-top: 10px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 30px 0; }}
  .stat-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; text-align: center; }}
  .stat-card .value {{ font-size: 1.8em; font-weight: 700; color: var(--accent); }}
  .stat-card .label {{ color: var(--text2); font-size: 0.85em; margin-top: 5px; }}
  section {{ margin: 40px 0; }}
  section h2 {{ font-size: 1.6em; font-weight: 400; border-left: 4px solid var(--accent); padding-left: 15px; margin-bottom: 20px; }}
  section h3 {{ font-size: 1.2em; font-weight: 400; color: var(--text2); margin: 20px 0 10px; }}
  .chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin: 20px 0; overflow: hidden; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 768px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  .data-table th {{ background: var(--accent2); color: var(--text); padding: 12px 15px; text-align: left; font-weight: 600; }}
  .data-table td {{ padding: 10px 15px; border-bottom: 1px solid var(--border); }}
  .data-table tr:hover td {{ background: rgba(233, 69, 96, 0.1); }}
  .insight {{ background: linear-gradient(135deg, #1a1a2e, #16213e); border-left: 4px solid var(--accent); padding: 15px 20px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
  .insight strong {{ color: var(--accent); }}
  .key-findings {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
  .finding {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  .finding .category {{ color: var(--accent); font-weight: 600; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; }}
  .finding .winner {{ font-size: 1.2em; margin: 8px 0; }}
  .finding .detail {{ color: var(--text2); font-size: 0.9em; }}
  footer {{ text-align: center; padding: 40px; color: var(--text2); font-size: 0.85em; border-top: 1px solid var(--border); margin-top: 60px; }}
  .plotly-graph-div {{ background: transparent !important; }}
  .js-plotly-plot .plotly .main-svg {{ background: transparent !important; }}
</style>
</head>
<body>

<header>
  <h1><span>Maqbool Fida Husain</span></h1>
  <h1 style="font-size:1.4em; margin-top:5px;">Auction Market Analysis</h1>
  <p>Christie's &amp; Sotheby's | {stats['years_range']} | {stats['total_lots']} lots analyzed</p>
</header>

<div class="container">

<!-- ── Overview Stats ──────────────────────────────────────────────── -->
<section>
  <h2>Market Overview</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="value">{stats['total_lots']}</div><div class="label">Total Lots</div></div>
    <div class="stat-card"><div class="value">{stats['total_sold']}</div><div class="label">Sold</div></div>
    <div class="stat-card"><div class="value">{stats['sell_through']}</div><div class="label">Sell-Through Rate</div></div>
    <div class="stat-card"><div class="value">{stats['total_revenue']}</div><div class="label">Total Revenue</div></div>
    <div class="stat-card"><div class="value">{stats['avg_price']}</div><div class="label">Average Price</div></div>
    <div class="stat-card"><div class="value">{stats['median_price']}</div><div class="label">Median Price</div></div>
    <div class="stat-card"><div class="value">{stats['max_price']}</div><div class="label">Highest Sale</div></div>
    <div class="stat-card"><div class="value">{stats['min_price']}</div><div class="label">Lowest Sale</div></div>
  </div>
</section>

<!-- ── Price Evolution ─────────────────────────────────────────────── -->
<section>
  <h2>Price Evolution Year over Year</h2>
  <div class="chart-card">{charts['price_evolution']}</div>
  <div class="chart-card">{yearly_table_html}</div>
  <div class="chart-row">
    <div class="chart-card">{charts['sales_sellthrough']}</div>
    <div class="chart-card">{charts['lots_per_year']}</div>
  </div>
</section>

<!-- ── Top Sales ───────────────────────────────────────────────────── -->
<section>
  <h2>Top 15 Most Expensive Works</h2>
  <div class="chart-card">{top15_html}</div>
</section>

<!-- ── What Types Do Best ──────────────────────────────────────────── -->
<section>
  <h2>What Types Perform Best?</h2>

  <div class="key-findings">
    <div class="finding">
      <div class="category">Best Medium</div>
      <div class="winner">{best_medium['Medium']}</div>
      <div class="detail">Avg: ${best_medium['Mean']:,.0f} | n={int(best_medium['Count'])}</div>
    </div>
    <div class="finding">
      <div class="category">Best Theme</div>
      <div class="winner">{best_theme['Theme']}</div>
      <div class="detail">Avg: ${best_theme['Mean']:,.0f} | n={int(best_theme['Count'])}</div>
    </div>
    <div class="finding">
      <div class="category">Best Decade</div>
      <div class="winner">{best_decade['Decade']}</div>
      <div class="detail">Avg: ${best_decade['Mean']:,.0f} | n={int(best_decade['Count'])}</div>
    </div>
    <div class="finding">
      <div class="category">Best Size</div>
      <div class="winner">{best_size['Size']}</div>
      <div class="detail">Avg: ${best_size['Mean']:,.0f} | n={int(best_size['Count'])}</div>
    </div>
  </div>

  <h3>By Medium</h3>
  <div class="chart-row">
    <div class="chart-card">{charts['medium_price']}</div>
    <div class="chart-card">{charts['box_medium']}</div>
  </div>
  <div class="chart-row">
    <div class="chart-card">{charts['medium_pie']}</div>
    <div class="chart-card">{charts['st_medium']}</div>
  </div>

  <h3>By Theme / Subject Matter</h3>
  <div class="chart-row">
    <div class="chart-card">{charts['theme_price']}</div>
    <div class="chart-card">{charts['theme_pie']}</div>
  </div>
  <div class="chart-card">{charts['st_theme']}</div>

  <h3>By Size of Work</h3>
  <div class="chart-row">
    <div class="chart-card">{charts['size_price']}</div>
    <div class="chart-card">{charts['scatter_size_price']}</div>
  </div>

  <h3>By Decade of Creation</h3>
  <div class="chart-row">
    <div class="chart-card">{charts['decade_price']}</div>
    <div class="chart-card">{charts['heatmap']}</div>
  </div>

  <h3>By Year Created</h3>
  <div class="chart-card">{charts['scatter_year_price']}</div>

  <h3>Signed vs. Unsigned</h3>
  <div class="chart-card">{charts['signed']}</div>
</section>

<!-- ── Estimate Accuracy ──────────────────────────────────────────── -->
<section>
  <h2>How Good Are Auction Estimates?</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="value">{stats['avg_deviation']}</div><div class="label">Avg Estimate Deviation</div></div>
    <div class="stat-card"><div class="value">{stats['pct_over']}</div><div class="label">Beat High Estimate</div></div>
    <div class="stat-card"><div class="value">{stats['pct_within']}</div><div class="label">Within Estimate</div></div>
    <div class="stat-card"><div class="value">{stats['pct_under']}</div><div class="label">Below Low Estimate</div></div>
  </div>
  <div class="chart-row">
    <div class="chart-card">{charts['estimate_accuracy']}</div>
    <div class="chart-card">{charts['estimate_outcome']}</div>
  </div>
</section>

<!-- ── Sale Type ──────────────────────────────────────────────────── -->
<section>
  <h2>Auction Type</h2>
  <div class="chart-card" style="max-width:500px;">{charts['sale_type']}</div>
</section>

</div>

<footer>
  M.F. Husain Auction Market Analysis | Data: Christie's &amp; Sotheby's ({stats['years_range']}) | Generated with Python + Plotly
</footer>

</body>
</html>"""

with open('husain_report.html', 'w') as f:
    f.write(html)

print("Report written to husain_report.html")
print(f"File size: {len(html) / 1024:.0f} KB")
