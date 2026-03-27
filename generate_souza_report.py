"""Generate a comprehensive HTML report for F.N. Souza auction analysis."""

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
mehta = pd.read_csv('data/raw/lots_souza_clean.csv')
mehta['auction_date'] = pd.to_datetime(mehta['auction_date'])
sold = mehta[mehta['is_sold'] == True].copy()
unsold = mehta[mehta['is_sold'] == False].copy()

# ── Unified price: hammer price for sold, estimate midpoint for unsold ──────
mehta['estimate_avg'] = mehta[['estimate_low_usd', 'estimate_high_usd']].mean(axis=1)
mehta['price_or_estimate'] = mehta['hammer_price_usd'].fillna(mehta['estimate_avg'])
mehta['price_label'] = mehta['is_sold'].map({True: 'Sold (Hammer)', False: 'Unsold (Est. Mid)'})

# estimate columns computed after sold is finalized below

# ── Clean medium ────────────────────────────────────────────────────────────
MEDIUM_MAP = {
    'oil_on_canvas': 'Oil on Canvas', 'acrylic_on_canvas': 'Acrylic on Canvas',
    'works_on_paper': 'Works on Paper', 'print': 'Print',
    'other': 'Other',
}
mehta['medium_clean'] = mehta['medium_category'].map(MEDIUM_MAP).fillna('Other')
sold['medium_clean'] = sold['medium_category'].map(MEDIUM_MAP).fillna('Other')

# ── Theme: already computed in clean data, copy to sold ─────────────────────
if 'theme' not in mehta.columns:
    mehta['theme'] = 'Other'
sold['theme'] = sold['theme'] if 'theme' in sold.columns else 'Other'

# ── Derived columns if missing ──────────────────────────────────────────────
if 'auction_year' not in mehta.columns:
    mehta['auction_year'] = mehta['auction_date'].dt.year
if 'surface_area_cm2' not in mehta.columns:
    mehta['surface_area_cm2'] = mehta['height_cm'] * mehta['width_cm']
sold = mehta[mehta['is_sold'] == True].copy()  # refresh sold after all columns added
sold['estimate_avg'] = sold[['estimate_low_usd', 'estimate_high_usd']].mean(axis=1)
sold['estimate_deviation'] = abs((sold['hammer_price_usd'] - sold['estimate_avg']) / sold['hammer_price_usd'])
sold['over_estimate'] = (sold['hammer_price_usd'] > sold['estimate_high_usd']).astype(int)
sold['under_estimate'] = (sold['hammer_price_usd'] < sold['estimate_low_usd']).astype(int)
sold['within_estimate'] = ((sold['hammer_price_usd'] >= sold['estimate_low_usd']) &
                           (sold['hammer_price_usd'] <= sold['estimate_high_usd'])).astype(int)

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

mehta['size_bucket'] = mehta['surface_area_cm2'].apply(size_bucket)
sold['size_bucket'] = sold['surface_area_cm2'].apply(size_bucket)

# ── Decade buckets ──────────────────────────────────────────────────────────
def decade_bucket(year):
    if pd.isna(year):
        return 'Unknown'
    return f"{int(year // 10 * 10)}s"

mehta['decade'] = mehta['year_created'].apply(decade_bucket)
sold['decade'] = sold['year_created'].apply(decade_bucket)

# ═══════════════════════════════════════════════════════════════════════════
#  BUILD CHARTS
# ═══════════════════════════════════════════════════════════════════════════
charts = {}

_first_chart = True
def chart_html(fig):
    """Convert figure to HTML fragment, embedding plotly.js only in the first chart."""
    global _first_chart
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    html = fig.to_html(full_html=False, include_plotlyjs=_first_chart)
    _first_chart = False
    return html

# ── 1. Price/Estimate Evolution Year over Year (ALL lots) ──────────────────
yearly_all = mehta.groupby('auction_year')['price_or_estimate'].agg(
    ['mean', 'median', 'min', 'max', 'count']
).reset_index()
yearly_all.columns = ['Year', 'Mean', 'Median', 'Low', 'High', 'Count']

# Sold-only yearly (may have gaps)
yearly_sold = sold.groupby('auction_year')['hammer_price_usd'].agg(
    ['mean', 'median', 'min', 'max', 'count']
).reset_index()
yearly_sold.columns = ['Year', 'Mean', 'Median', 'Low', 'High', 'Count']

fig = go.Figure()
fig.add_trace(go.Scatter(x=yearly_all['Year'], y=yearly_all['High'], mode='lines+markers',
                         name='High (all lots)', line=dict(color='#e74c3c', width=2)))
fig.add_trace(go.Scatter(x=yearly_all['Year'], y=yearly_all['Mean'], mode='lines+markers',
                         name='Mean (all lots)', line=dict(color='#3498db', width=3)))
fig.add_trace(go.Scatter(x=yearly_all['Year'], y=yearly_all['Median'], mode='lines+markers',
                         name='Median (all lots)', line=dict(color='#2ecc71', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=yearly_all['Year'], y=yearly_all['Low'], mode='lines+markers',
                         name='Low (all lots)', line=dict(color='#f39c12', width=2)))
# Overlay sold-only markers
if len(yearly_sold) > 0:
    fig.add_trace(go.Scatter(x=yearly_sold['Year'], y=yearly_sold['Mean'], mode='markers',
                             name='Mean (sold only)', marker=dict(color='#e94560', size=12, symbol='diamond')))
fig.add_trace(go.Bar(x=yearly_all['Year'], y=yearly_all['Count'], name='# Lots',
                     yaxis='y2', opacity=0.15, marker_color='#95a5a6'))
fig.update_layout(
    title='Value Evolution Year over Year<br><sub>Sold lots use hammer price; unsold lots use estimate midpoint</sub>',
    xaxis_title='Auction Year', yaxis_title='Price / Estimate (USD)',
    yaxis2=dict(title='# Lots', overlaying='y', side='right'),
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    height=550,
)
charts['price_evolution'] = chart_html(fig)

# ── 2. Total Estimated Value & Sell-Through by Year ─────────────────────────
lots_yr = mehta.groupby('auction_year').size().reset_index(name='total')
sold_yr = sold.groupby('auction_year').size().reset_index(name='sold')
merged_yr = lots_yr.merge(sold_yr, on='auction_year', how='left').fillna(0)
merged_yr['sell_through'] = merged_yr['sold'] / merged_yr['total'] * 100
total_val_yr = mehta.groupby('auction_year')['price_or_estimate'].sum().reset_index(name='total_value')
merged_yr = merged_yr.merge(total_val_yr, on='auction_year', how='left').fillna(0)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=merged_yr['auction_year'], y=merged_yr['total_value'],
                     name='Total Value (USD)', marker_color='#3498db'), secondary_y=False)
fig.add_trace(go.Scatter(x=merged_yr['auction_year'], y=merged_yr['sell_through'],
                         name='Sell-Through %', mode='lines+markers',
                         line=dict(color='#e74c3c', width=3)), secondary_y=True)
fig.update_layout(title='Total Market Value & Sell-Through Rate by Year<br><sub>Value = hammer (sold) + estimate mid (unsold)</sub>',
                  xaxis_title='Year', height=450,
                  legend=dict(orientation='h', yanchor='bottom', y=1.02))
fig.update_yaxes(title_text='Total Value (USD)', secondary_y=False)
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
med_counts = mehta['medium_clean'].value_counts()
fig = go.Figure(go.Pie(labels=med_counts.index, values=med_counts.values,
                       textinfo='label+percent', hole=0.3))
fig.update_layout(title='Distribution of Mediums (All Lots)', height=450)
charts['medium_pie'] = chart_html(fig)

# ── 5. Value by Medium (all lots, using price_or_estimate) ──────────────────
med_price = mehta.groupby('medium_clean')['price_or_estimate'].agg(['mean', 'median', 'count']).reset_index()
med_price.columns = ['Medium', 'Mean', 'Median', 'Count']
med_price = med_price.sort_values('Mean', ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(y=med_price['Medium'], x=med_price['Mean'], orientation='h',
                     name='Mean Value', marker_color='#3498db',
                     text=[f"${v:,.0f} (n={c})" for v, c in zip(med_price['Mean'], med_price['Count'])],
                     textposition='outside'))
fig.add_trace(go.Bar(y=med_price['Medium'], x=med_price['Median'], orientation='h',
                     name='Median Value', marker_color='#2ecc71'))
fig.update_layout(title='Value by Medium (All Lots)<br><sub>Hammer price for sold; estimate midpoint for unsold</sub>',
                  barmode='group', xaxis_title='Value (USD)', height=400, margin=dict(l=120))
charts['medium_price'] = chart_html(fig)

# ── 6. Value by Theme (all lots) ───────────────────────────────────────────
theme_price = mehta.groupby('theme')['price_or_estimate'].agg(['mean', 'median', 'count']).reset_index()
theme_price.columns = ['Theme', 'Mean', 'Median', 'Count']
theme_price = theme_price.sort_values('Mean', ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(y=theme_price['Theme'], x=theme_price['Mean'], orientation='h',
                     name='Mean Value', marker_color='#9b59b6',
                     text=[f"${v:,.0f} (n={c})" for v, c in zip(theme_price['Mean'], theme_price['Count'])],
                     textposition='outside'))
fig.add_trace(go.Bar(y=theme_price['Theme'], x=theme_price['Median'], orientation='h',
                     name='Median Value', marker_color='#e67e22'))
fig.update_layout(title='Value by Theme/Subject (All Lots)', barmode='group',
                  xaxis_title='Value (USD)', height=450, margin=dict(l=180))
charts['theme_price'] = chart_html(fig)

# ── 7. Theme distribution (all lots) ───────────────────────────────────────
theme_counts = mehta['theme'].value_counts()
fig = go.Figure(go.Pie(labels=theme_counts.index, values=theme_counts.values,
                       textinfo='label+percent', hole=0.3))
fig.update_layout(title='Theme/Subject Distribution (All Lots)', height=450)
charts['theme_pie'] = chart_html(fig)

# ── 8. Value by Size Bucket (all lots) ──────────────────────────────────────
size_order = ['Small (<1000 cm²)', 'Medium (1000-3000 cm²)', 'Large (3000-6000 cm²)', 'Very Large (>6000 cm²)']
size_price = mehta[mehta['size_bucket'] != 'Unknown'].groupby('size_bucket')['price_or_estimate'].agg(
    ['mean', 'median', 'count']).reset_index()
size_price.columns = ['Size', 'Mean', 'Median', 'Count']
size_price['Size'] = pd.Categorical(size_price['Size'], categories=size_order, ordered=True)
size_price = size_price.sort_values('Size')

fig = go.Figure()
fig.add_trace(go.Bar(x=size_price['Size'], y=size_price['Mean'], name='Mean Value',
                     marker_color='#1abc9c',
                     text=[f"${v:,.0f}<br>n={c}" for v, c in zip(size_price['Mean'], size_price['Count'])],
                     textposition='outside'))
fig.add_trace(go.Bar(x=size_price['Size'], y=size_price['Median'], name='Median Value',
                     marker_color='#16a085'))
fig.update_layout(title='Value by Size of Work (All Lots)', barmode='group',
                  xaxis_title='Size Category', yaxis_title='Value (USD)', height=450)
charts['size_price'] = chart_html(fig)

# ── 9. Value by Decade Created (all lots) ───────────────────────────────────
decade_price = mehta[mehta['decade'] != 'Unknown'].groupby('decade')['price_or_estimate'].agg(
    ['mean', 'median', 'count']).reset_index()
decade_price.columns = ['Decade', 'Mean', 'Median', 'Count']
decade_price = decade_price.sort_values('Decade')

fig = go.Figure()
fig.add_trace(go.Bar(x=decade_price['Decade'], y=decade_price['Mean'], name='Mean Value',
                     marker_color='#e74c3c',
                     text=[f"${v:,.0f}<br>n={c}" for v, c in zip(decade_price['Mean'], decade_price['Count'])],
                     textposition='outside'))
fig.add_trace(go.Bar(x=decade_price['Decade'], y=decade_price['Median'], name='Median Value',
                     marker_color='#c0392b'))
fig.update_layout(title='Value by Decade of Creation (All Lots)', barmode='group',
                  xaxis_title='Decade Created', yaxis_title='Value (USD)', height=450)
charts['decade_price'] = chart_html(fig)

# ── 10. Scatter: Value vs Surface Area (all lots, color by sold/unsold) ─────
scatter_data = mehta.dropna(subset=['surface_area_cm2']).copy()
fig = px.scatter(scatter_data,
                 x='surface_area_cm2', y='price_or_estimate',
                 color='price_label', symbol='price_label',
                 hover_data=['title', 'auction_year', 'medium_clean'],
                 trendline='ols',
                 title='Value vs. Surface Area (Sold & Unsold)')
fig.update_layout(xaxis_title='Surface Area (cm²)', yaxis_title='Value (USD)', height=500)
charts['scatter_size_price'] = chart_html(fig)

# ── 11. Scatter: Year Created vs Value (all lots) ──────────────────────────
scatter_yr = mehta.dropna(subset=['year_created']).copy()
fig = px.scatter(scatter_yr,
                 x='year_created', y='price_or_estimate',
                 color='price_label', symbol='medium_clean',
                 hover_data=['title'],
                 title='Year Created vs. Value (Sold & Unsold)')
fig.update_layout(xaxis_title='Year Created', yaxis_title='Value (USD)', height=500)
charts['scatter_year_price'] = chart_html(fig)

# ── 12. Estimate vs Hammer (sold only) ─────────────────────────────────────
if len(sold) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sold['estimate_avg'], y=sold['hammer_price_usd'],
                             mode='markers+text', marker=dict(size=14, opacity=0.8, color='#3498db'),
                             text=sold['title'], textposition='top center',
                             name='Sold Lots'))
    max_val = max(sold['estimate_avg'].max(), sold['hammer_price_usd'].max()) * 1.2
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                             line=dict(dash='dash', color='red'), name='Perfect Estimate'))
    fig.update_layout(title="Estimate Accuracy: Estimate vs. Hammer Price (Sold Only)",
                      xaxis_title='Estimate Average (USD)', yaxis_title='Hammer Price (USD)', height=500)
    charts['estimate_accuracy'] = chart_html(fig)

    outcome_counts = pd.Series({
        'Over Estimate': int(sold['over_estimate'].sum()),
        'Within Estimate': int(sold['within_estimate'].sum()),
        'Under Estimate': int(sold['under_estimate'].sum()),
    })
    fig = go.Figure(go.Pie(labels=outcome_counts.index, values=outcome_counts.values,
                           marker_colors=['#2ecc71', '#3498db', '#e74c3c'],
                           textinfo='label+percent+value', hole=0.4))
    fig.update_layout(title='How Often Do Lots Beat/Meet/Miss Estimates?', height=400)
    charts['estimate_outcome'] = chart_html(fig)

# ── 13. Estimate Range Visualization (all lots) ────────────────────────────
# Show each lot as a bar from estimate_low to estimate_high, with hammer price as a marker
all_lots = mehta.sort_values('price_or_estimate', ascending=True).reset_index(drop=True)
fig = go.Figure()
for i, row in all_lots.iterrows():
    color = '#2ecc71' if row['is_sold'] else '#e74c3c'
    fig.add_trace(go.Scatter(
        x=[row['estimate_low_usd'], row['estimate_high_usd']],
        y=[row['title'], row['title']],
        mode='lines', line=dict(color='#555', width=6),
        showlegend=False, hoverinfo='skip'
    ))
    if row['is_sold'] and pd.notna(row['hammer_price_usd']):
        fig.add_trace(go.Scatter(
            x=[row['hammer_price_usd']], y=[row['title']],
            mode='markers', marker=dict(color='#2ecc71', size=12, symbol='diamond'),
            name='Hammer Price' if i == all_lots[all_lots['is_sold']].index[0] else None,
            showlegend=bool(i == all_lots[all_lots['is_sold']].index[0]),
            hovertext=f"${row['hammer_price_usd']:,.0f}",
        ))
    fig.add_trace(go.Scatter(
        x=[row['estimate_avg']], y=[row['title']],
        mode='markers', marker=dict(color='#3498db' if not row['is_sold'] else '#f39c12',
                                     size=8, symbol='circle'),
        showlegend=False,
        hovertext=f"Est. Mid: ${row['estimate_avg']:,.0f}",
    ))
fig.update_layout(
    title='All Lots: Estimate Range & Hammer Price<br><sub>Gray bar = estimate range | Diamond = hammer price | Circle = estimate midpoint</sub>',
    xaxis_title='Value (USD)', height=max(450, len(all_lots) * 40),
    yaxis=dict(autorange='reversed'), margin=dict(l=250),
)
charts['lot_ranges'] = chart_html(fig)

# ── 14. Sell-Through by Medium ──────────────────────────────────────────────
med_total = mehta.groupby('medium_clean').size().reset_index(name='total')
med_sold_ct = sold.groupby('medium_clean').size().reset_index(name='sold')
med_st = med_total.merge(med_sold_ct, on='medium_clean', how='left').fillna(0)
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
th_total = mehta.groupby('theme').size().reset_index(name='total')
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
                  height=400, margin=dict(l=180), xaxis_range=[0, 110])
charts['st_theme'] = chart_html(fig)

# ── 16. Sale Type (Live vs Online) ─────────────────────────────────────────
sale_counts = mehta['sale_type'].value_counts()
fig = go.Figure(go.Pie(labels=sale_counts.index, values=sale_counts.values,
                       textinfo='label+percent+value', hole=0.3,
                       marker_colors=['#3498db', '#e67e22']))
fig.update_layout(title='Auction Type Distribution', height=400)
charts['sale_type'] = chart_html(fig)

# ── 17. All Lots Table ─────────────────────────────────────────────────────
all_table = mehta.sort_values('price_or_estimate', ascending=False)[
    ['title', 'medium_clean', 'hammer_price_usd', 'estimate_low_usd',
     'estimate_high_usd', 'surface_area_cm2', 'year_created',
     'auction_year', 'is_sold']].copy()
all_table['hammer_price_usd'] = all_table['hammer_price_usd'].apply(
    lambda x: f"${x:,.0f}" if pd.notna(x) else '—')
all_table['estimate_low_usd'] = all_table['estimate_low_usd'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else 'N/A')
all_table['estimate_high_usd'] = all_table['estimate_high_usd'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else 'N/A')
all_table['surface_area_cm2'] = all_table['surface_area_cm2'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
all_table['year_created'] = all_table['year_created'].apply(lambda x: f"{int(x)}" if pd.notna(x) else 'N/A')
all_table['is_sold'] = all_table['is_sold'].map({True: 'Sold', False: 'Unsold'})
all_table.columns = ['Title', 'Medium', 'Hammer Price', 'Est. Low', 'Est. High',
                      'Area (cm²)', 'Year Created', 'Year Auctioned', 'Status']

# ── 18. Signed vs Unsigned (all lots, using price_or_estimate) ──────────────
signed_data = []
for label, mask in [('Signed', mehta['is_signed'] == True), ('Unsigned', mehta['is_signed'] == False)]:
    subset = mehta[mask]['price_or_estimate']
    if len(subset) > 0:
        signed_data.append({'Status': label, 'Mean': subset.mean(), 'Median': subset.median(), 'Count': len(subset)})
signed_df = pd.DataFrame(signed_data)

if len(signed_df) > 0:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=signed_df['Status'], y=signed_df['Mean'], name='Mean',
                         marker_color='#3498db',
                         text=[f"${v:,.0f} (n={c})" for v, c in zip(signed_df['Mean'], signed_df['Count'])],
                         textposition='outside'))
    fig.add_trace(go.Bar(x=signed_df['Status'], y=signed_df['Median'], name='Median',
                         marker_color='#2ecc71'))
    fig.update_layout(title='Signed vs. Unsigned Value Comparison (All Lots)', barmode='group',
                      yaxis_title='Value (USD)', height=400)
    charts['signed'] = chart_html(fig)

# ── 19. Box plot: Value distribution by medium (all lots) ───────────────────
fig = px.box(mehta, x='medium_clean', y='price_or_estimate', color='price_label',
             title='Value Distribution by Medium (Sold & Unsold)', points='all',
             hover_data=['title', 'auction_year'])
fig.update_layout(xaxis_title='Medium', yaxis_title='Value (USD)', height=500)
charts['box_medium'] = chart_html(fig)

# ── 20. Heatmap: Medium x Decade (all lots) ────────────────────────────────
mehta_known = mehta[mehta['decade'] != 'Unknown'].copy()
if len(mehta_known) > 0:
    pivot = mehta_known.pivot_table(values='price_or_estimate', index='medium_clean',
                                    columns='decade', aggfunc='mean')
    pivot_count = mehta_known.pivot_table(values='price_or_estimate', index='medium_clean',
                                          columns='decade', aggfunc='count').fillna(0)
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
        colorbar_title='Avg Value (USD)',
    ))
    fig.update_layout(title='Average Value Heatmap: Medium x Decade (All Lots)',
                      xaxis_title='Decade Created', yaxis_title='Medium', height=400)
    charts['heatmap'] = chart_html(fig)

# ═══════════════════════════════════════════════════════════════════════════
#  REGRESSION & VALUATION MODEL
# ═══════════════════════════════════════════════════════════════════════════
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

CURRENT_YEAR = 2026

# --- 1. Hedonic regression: log(price) ~ year + medium + log(area) ---
reg_data = sold[['auction_year', 'hammer_price_usd', 'title', 'medium_clean',
                  'surface_area_cm2']].dropna().copy()
reg_data['log_price'] = np.log(reg_data['hammer_price_usd'].clip(lower=1))
reg_data['log_area'] = np.log(reg_data['surface_area_cm2'].clip(lower=1))
reg_data['year_numeric'] = reg_data['auction_year'].astype(float)

# Encode medium as dummies
medium_dummies = pd.get_dummies(reg_data['medium_clean'], prefix='med', drop_first=True)
X_full = pd.concat([reg_data[['year_numeric', 'log_area']], medium_dummies], axis=1)
y = reg_data['log_price']

model = LinearRegression().fit(X_full, y)
reg_data['predicted_log'] = model.predict(X_full)
reg_data['residual'] = y - reg_data['predicted_log']
resid_std = reg_data['residual'].std()

# Extract year coefficient for appreciation rate
year_coef = model.coef_[0]  # coefficient on year_numeric
annual_appreciation_pct = (np.exp(year_coef) - 1) * 100

# Also run simple regression for trend line chart
slope_simple, intercept_simple, r_simple, p_simple, se_simple = sp_stats.linregress(
    reg_data['year_numeric'], reg_data['log_price']
)
r_squared_simple = r_simple ** 2

# Full model R²
r_squared_full = model.score(X_full, y)

# Regression stats dict
reg_stats = {
    'year_coef': year_coef,
    'r_squared_simple': f"{r_squared_simple:.3f}",
    'r_squared_full': f"{r_squared_full:.3f}",
    'p_value': f"{p_simple:.4f}" if p_simple >= 0.0001 else f"{p_simple:.2e}",
    'annual_appreciation': f"{annual_appreciation_pct:+.1f}%",
    'resid_std': f"{resid_std:.2f}",
    'n': len(reg_data),
}

# --- Chart 1: Scatter + trend line + forecast ---
year_range = np.arange(reg_data['year_numeric'].min(), 2036)
# Use the median lot profile (median area, most common medium) for the trend line
median_log_area = reg_data['log_area'].median()
# Build feature vector for trend line prediction
trend_X = pd.DataFrame({'year_numeric': year_range, 'log_area': median_log_area})
for col in medium_dummies.columns:
    trend_X[col] = 0  # base medium category
trend_fitted = model.predict(trend_X)
trend_price = np.exp(trend_fitted)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=reg_data['year_numeric'], y=reg_data['hammer_price_usd'],
    mode='markers', name='Sold Lots',
    marker=dict(size=10, color='#3498db', opacity=0.7),
    text=reg_data['title'], hoverinfo='text+x+y'
))
# Historical fitted line
hist_mask = year_range <= reg_data['year_numeric'].max()
fig.add_trace(go.Scatter(
    x=year_range[hist_mask], y=trend_price[hist_mask],
    mode='lines', name=f'Trend ({annual_appreciation_pct:+.1f}%/yr)',
    line=dict(color='#e94560', width=3)
))
# Forecast line
forecast_mask = year_range >= reg_data['year_numeric'].max()
fig.add_trace(go.Scatter(
    x=year_range[forecast_mask], y=trend_price[forecast_mask],
    mode='lines', name='Forecast',
    line=dict(color='#e94560', width=3, dash='dot')
))
# 95% prediction interval
upper_pi = np.exp(trend_fitted + 1.96 * resid_std)
lower_pi = np.exp(trend_fitted - 1.96 * resid_std)
fig.add_trace(go.Scatter(
    x=np.concatenate([year_range, year_range[::-1]]),
    y=np.concatenate([upper_pi, lower_pi[::-1]]),
    fill='toself', fillcolor='rgba(233,69,96,0.1)',
    line=dict(color='rgba(0,0,0,0)'), showlegend=True, name='95% Prediction Interval'
))
fig.update_layout(
    title=f'Price Appreciation Model (Median-Profile Lot)<br><sub>Hedonic regression: R²={r_squared_full:.3f}, '
          f'Year coeff appreciation: {annual_appreciation_pct:+.1f}%/yr, '
          f'Simple R²={r_squared_simple:.3f}, p={reg_stats["p_value"]}</sub>',
    xaxis_title='Year', yaxis_title='Hammer Price (USD)',
    yaxis_type='log', height=600,
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
)
charts['regression'] = chart_html(fig)

# --- 2. Predicted 2026 value for every lot ---
# Use the hedonic model: predict what each lot's features would yield in 2026
# Keep each lot's residual (its lot-specific premium/discount) and re-predict at 2026
sold_val = sold.copy()
sold_val['log_price'] = np.log(sold_val['hammer_price_usd'].clip(lower=1))
sold_val['log_area'] = np.log(sold_val['surface_area_cm2'].clip(lower=1))

# Build features at original year
X_orig = pd.concat([
    sold_val[['auction_year', 'log_area']].rename(columns={'auction_year': 'year_numeric'}),
    pd.get_dummies(sold_val['medium_clean'], prefix='med', drop_first=True).reindex(
        columns=medium_dummies.columns, fill_value=0)
], axis=1)
sold_val['predicted_log_orig'] = model.predict(X_orig)
sold_val['residual'] = sold_val['log_price'] - sold_val['predicted_log_orig']

# Build features at 2026 (same lot, just year changes)
X_2026 = X_orig.copy()
X_2026['year_numeric'] = CURRENT_YEAR
sold_val['predicted_log_2026'] = model.predict(X_2026) + sold_val['residual']

# Convert back from log and cap at reasonable multiple of max historical sale
max_historical = sold_val['hammer_price_usd'].max()
sold_val['predicted_2026'] = np.exp(sold_val['predicted_log_2026']).clip(upper=max_historical * 1.5)
sold_val['gain_loss_usd'] = sold_val['predicted_2026'] - sold_val['hammer_price_usd']
sold_val['gain_loss_pct'] = (sold_val['gain_loss_usd'] / sold_val['hammer_price_usd']) * 100

# Chart: Predicted vs actual (top 30 by predicted value)
top_val = sold_val.nlargest(30, 'predicted_2026').copy()
top_val = top_val.sort_values('predicted_2026', ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=top_val['title'], x=top_val['hammer_price_usd'], orientation='h',
    name='Sale Price', marker_color='#3498db',
    text=[f"${v:,.0f}" for v in top_val['hammer_price_usd']], textposition='inside',
))
fig.add_trace(go.Bar(
    y=top_val['title'], x=top_val['predicted_2026'], orientation='h',
    name='Est. 2026 Value', marker_color='#2ecc71',
    text=[f"${v:,.0f}" for v in top_val['predicted_2026']], textposition='inside',
))
fig.update_layout(
    title='Historical Sale Price vs. Estimated 2026 Value (Top 30)',
    barmode='group', xaxis_title='USD', height=max(500, len(top_val) * 28),
    margin=dict(l=250), legend=dict(orientation='h', yanchor='bottom', y=1.02),
)
charts['predicted_vs_actual'] = chart_html(fig)

# Chart: Gain/loss (all lots sorted by year)
sold_sorted = sold_val.sort_values('auction_year').copy()
colors = ['#2ecc71' if g > 0 else '#e74c3c' for g in sold_sorted['gain_loss_pct']]
fig = go.Figure(go.Bar(
    x=sold_sorted['title'] + ' (' + sold_sorted['auction_year'].astype(str) + ')',
    y=sold_sorted['gain_loss_pct'],
    marker_color=colors,
    text=[f"{v:+.0f}%" for v in sold_sorted['gain_loss_pct']],
    textposition='outside',
))
fig.update_layout(
    title='Estimated Gain/Loss if Held to 2026 (%)',
    xaxis_title='', yaxis_title='Gain/Loss %',
    height=600, xaxis_tickangle=-45, margin=dict(b=200),
)
charts['gain_loss'] = chart_html(fig)

# --- 3. Forecast appreciation table ---
forecast_years = [2026, 2027, 2028, 2029, 2030, 2035]
# Use actual median hammer price as baseline
current_median = sold_val[sold_val['auction_year'] >= 2020]['hammer_price_usd'].median()
if pd.isna(current_median):
    current_median = sold_val['hammer_price_usd'].median()
forecast_rows = []
for yr in forecast_years:
    yrs_from_now = yr - CURRENT_YEAR
    multiplier = np.exp(year_coef * yrs_from_now)
    median_val = current_median * multiplier
    total_appr = (multiplier - 1) * 100
    forecast_rows.append({
        'Year': yr,
        'Multiplier': f"{multiplier:.2f}x",
        'Projected Median Value': f"${median_val:,.0f}",
        'Cumulative Appreciation': f"{total_appr:+.1f}%",
    })
forecast_df = pd.DataFrame(forecast_rows)

# --- 4. Per-medium regression ---
medium_reg_rows = []
for med in sold_val['medium_clean'].unique():
    med_data = sold_val[sold_val['medium_clean'] == med]
    if len(med_data) >= 3:
        s, i, r, p, se = sp_stats.linregress(
            med_data['auction_year'].astype(float), np.log(med_data['hammer_price_usd'].clip(lower=1))
        )
        ann_pct = (np.exp(s) - 1) * 100
        medium_reg_rows.append({
            'Medium': med,
            'Annual Appr.': f"{ann_pct:+.1f}%",
            'R²': f"{r**2:.3f}",
            'p-value': f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}",
            'n': len(med_data),
        })
medium_reg_df = pd.DataFrame(medium_reg_rows).sort_values('Medium') if medium_reg_rows else pd.DataFrame()

# --- 5. Valuation table (all lots) ---
val_table = sold_val[['title', 'medium_clean', 'auction_year', 'hammer_price_usd',
                       'predicted_2026', 'gain_loss_usd', 'gain_loss_pct']].copy()
val_table = val_table.sort_values('predicted_2026', ascending=False)
val_table['hammer_price_usd'] = val_table['hammer_price_usd'].apply(lambda x: f"${x:,.0f}")
val_table['predicted_2026'] = val_table['predicted_2026'].apply(lambda x: f"${x:,.0f}")
val_table['gain_loss_usd'] = val_table['gain_loss_usd'].apply(lambda x: f"${x:+,.0f}")
val_table['gain_loss_pct'] = val_table['gain_loss_pct'].apply(lambda x: f"{x:+.1f}%")
val_table.columns = ['Title', 'Medium', 'Year Sold', 'Sale Price', 'Est. 2026 Value',
                      'Gain/Loss ($)', 'Gain/Loss (%)']

# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY STATS
# ═══════════════════════════════════════════════════════════════════════════
stats = {
    'total_lots': len(mehta),
    'total_sold': len(sold),
    'total_unsold': len(unsold),
    'sell_through': f"{len(sold)/len(mehta)*100:.1f}%",
    'avg_hammer': f"${sold['hammer_price_usd'].mean():,.0f}" if len(sold) > 0 else 'N/A',
    'median_hammer': f"${sold['hammer_price_usd'].median():,.0f}" if len(sold) > 0 else 'N/A',
    'max_hammer': f"${sold['hammer_price_usd'].max():,.0f}" if len(sold) > 0 else 'N/A',
    'avg_value': f"${mehta['price_or_estimate'].mean():,.0f}",
    'median_value': f"${mehta['price_or_estimate'].median():,.0f}",
    'max_value': f"${mehta['price_or_estimate'].max():,.0f}",
    'min_value': f"${mehta['price_or_estimate'].min():,.0f}",
    'total_value': f"${mehta['price_or_estimate'].sum():,.0f}",
    'avg_estimate': f"${mehta['estimate_avg'].mean():,.0f}",
    'years_range': f"{int(mehta['auction_year'].min())}-{int(mehta['auction_year'].max())}",
    'creation_range': f"{int(mehta['year_created'].dropna().min())}-{int(mehta['year_created'].dropna().max())}" if mehta['year_created'].notna().any() else 'N/A',
}

if len(sold) > 0:
    stats['avg_deviation'] = f"{sold['estimate_deviation'].mean()*100:.1f}%"
    stats['pct_over'] = f"{sold['over_estimate'].mean()*100:.0f}%"
    stats['pct_within'] = f"{sold['within_estimate'].mean()*100:.0f}%"
    stats['pct_under'] = f"{sold['under_estimate'].mean()*100:.0f}%"

# best performers
best_medium = med_price.sort_values('Mean', ascending=False).iloc[0]
best_theme = theme_price.sort_values('Mean', ascending=False).iloc[0]
best_decade = decade_price.sort_values('Mean', ascending=False).iloc[0] if len(decade_price) > 0 else None
best_size = size_price.sort_values('Mean', ascending=False).iloc[0] if len(size_price) > 0 else None

# ═══════════════════════════════════════════════════════════════════════════
#  BUILD HTML
# ═══════════════════════════════════════════════════════════════════════════
all_table_html = all_table.to_html(index=False, classes='data-table', border=0)

yearly_table = yearly_all.copy()
yearly_table['Mean'] = yearly_table['Mean'].apply(lambda x: f"${x:,.0f}")
yearly_table['Median'] = yearly_table['Median'].apply(lambda x: f"${x:,.0f}")
yearly_table['Low'] = yearly_table['Low'].apply(lambda x: f"${x:,.0f}")
yearly_table['High'] = yearly_table['High'].apply(lambda x: f"${x:,.0f}")
yearly_table.columns = ['Year', 'Mean', 'Median', 'Low', 'High', '# Lots']
yearly_table_html = yearly_table.to_html(index=False, classes='data-table', border=0)

# Build findings cards
findings_html = ""
if best_medium is not None:
    findings_html += f"""<div class="finding"><div class="category">Best Medium</div>
      <div class="winner">{best_medium['Medium']}</div>
      <div class="detail">Avg: ${best_medium['Mean']:,.0f} | n={int(best_medium['Count'])}</div></div>"""
if best_theme is not None:
    findings_html += f"""<div class="finding"><div class="category">Best Theme</div>
      <div class="winner">{best_theme['Theme']}</div>
      <div class="detail">Avg: ${best_theme['Mean']:,.0f} | n={int(best_theme['Count'])}</div></div>"""
if best_decade is not None:
    findings_html += f"""<div class="finding"><div class="category">Best Decade</div>
      <div class="winner">{best_decade['Decade']}</div>
      <div class="detail">Avg: ${best_decade['Mean']:,.0f} | n={int(best_decade['Count'])}</div></div>"""
if best_size is not None:
    findings_html += f"""<div class="finding"><div class="category">Best Size</div>
      <div class="winner">{best_size['Size']}</div>
      <div class="detail">Avg: ${best_size['Mean']:,.0f} | n={int(best_size['Count'])}</div></div>"""

# Estimate accuracy section (conditional)
estimate_section = ""
if len(sold) > 0 and 'estimate_accuracy' in charts:
    estimate_section = f"""
<section>
  <h2>How Good Are Auction Estimates?</h2>
  <div class="insight">Based on {len(sold)} sold lots only.</div>
  <div class="stats-grid">
    <div class="stat-card"><div class="value">{stats.get('avg_deviation','N/A')}</div><div class="label">Avg Estimate Deviation</div></div>
    <div class="stat-card"><div class="value">{stats.get('pct_over','N/A')}</div><div class="label">Beat High Estimate</div></div>
    <div class="stat-card"><div class="value">{stats.get('pct_within','N/A')}</div><div class="label">Within Estimate</div></div>
    <div class="stat-card"><div class="value">{stats.get('pct_under','N/A')}</div><div class="label">Below Low Estimate</div></div>
  </div>
  <div class="chart-row">
    <div class="chart-card">{charts['estimate_accuracy']}</div>
    <div class="chart-card">{charts.get('estimate_outcome','')}</div>
  </div>
</section>"""

signed_section = ""
if 'signed' in charts:
    signed_section = f"""<h3>Signed vs. Unsigned</h3>
  <div class="chart-card">{charts['signed']}</div>"""

heatmap_section = ""
if 'heatmap' in charts:
    heatmap_section = f"""<div class="chart-card">{charts['heatmap']}</div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>F.N. Souza - Auction Market Analysis</title>
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
  .note {{ color: var(--text2); font-size: 0.9em; font-style: italic; margin: 10px 0; }}
</style>
</head>
<body>

<header>
  <h1><span>Francis Newton Souza</span></h1>
  <h1 style="font-size:1.4em; margin-top:5px;">Auction Market Analysis</h1>
  <p>Christie's | {stats['years_range']} | {stats['total_lots']} lots analyzed</p>
</header>

<div class="container">

<div class="insight">
  <strong>Data source:</strong> {stats['total_lots']} lots scraped from Christie's auction records ({stats['years_range']}).
  {stats['total_sold']} sold lots with hammer prices. Where applicable, estimate midpoints supplement hammer prices.
</div>

<!-- ── Overview Stats ──────────────────────────────────────────────── -->
<section>
  <h2>Market Overview</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="value">{stats['total_lots']}</div><div class="label">Total Lots</div></div>
    <div class="stat-card"><div class="value">{stats['total_sold']}</div><div class="label">Sold</div></div>
    <div class="stat-card"><div class="value">{stats['total_unsold']}</div><div class="label">Unsold</div></div>
    <div class="stat-card"><div class="value">{stats['sell_through']}</div><div class="label">Sell-Through Rate</div></div>
    <div class="stat-card"><div class="value">{stats['avg_hammer']}</div><div class="label">Avg Hammer (Sold)</div></div>
    <div class="stat-card"><div class="value">{stats['avg_value']}</div><div class="label">Avg Value (All Lots)</div></div>
    <div class="stat-card"><div class="value">{stats['max_value']}</div><div class="label">Highest Value</div></div>
    <div class="stat-card"><div class="value">{stats['total_value']}</div><div class="label">Total Market Value</div></div>
  </div>
</section>

<!-- ── Price Evolution ─────────────────────────────────────────────── -->
<section>
  <h2>Value Evolution Year over Year</h2>
  <div class="chart-card">{charts['price_evolution']}</div>
  <div class="chart-card">{yearly_table_html}</div>
  <div class="chart-row">
    <div class="chart-card">{charts['sales_sellthrough']}</div>
    <div class="chart-card">{charts['lots_per_year']}</div>
  </div>
</section>

<!-- ── All Lots Detail ─────────────────────────────────────────────── -->
<section>
  <h2>All Lots at Auction</h2>
  <div class="chart-card">{charts['lot_ranges']}</div>
  <div class="chart-card">{all_table_html}</div>
</section>

<!-- ── What Types Do Best ──────────────────────────────────────────── -->
<section>
  <h2>What Types Perform Best?</h2>
  <p class="note">Values = hammer price (sold) or estimate midpoint (unsold)</p>

  <div class="key-findings">{findings_html}</div>

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
    {heatmap_section}
  </div>

  <h3>By Year Created</h3>
  <div class="chart-card">{charts['scatter_year_price']}</div>

  {signed_section}
</section>

{estimate_section}

<!-- ── Regression & Valuation ──────────────────────────────────────── -->
<section>
  <h2>Price Appreciation Model</h2>
  <div class="insight">
    <strong>Methodology:</strong> Hedonic log-linear regression: log(price) ~ year + medium + log(area).
    Controls for lot characteristics (size, medium) to isolate the time appreciation effect.
    2026 estimates are capped at 1.5x the historical max sale to avoid runaway extrapolation.
    The wide prediction interval reflects high variance in art prices — treat forecasts as directional, not precise.
  </div>

  <div class="stats-grid">
    <div class="stat-card"><div class="value">{reg_stats['annual_appreciation']}</div><div class="label">Annual Appreciation (year coeff)</div></div>
    <div class="stat-card"><div class="value">{reg_stats['r_squared_full']}</div><div class="label">R² (Hedonic Model)</div></div>
    <div class="stat-card"><div class="value">{reg_stats['r_squared_simple']}</div><div class="label">R² (Year Only)</div></div>
    <div class="stat-card"><div class="value">{reg_stats['p_value']}</div><div class="label">p-value (year)</div></div>
    <div class="stat-card"><div class="value">{reg_stats['n']}</div><div class="label">Observations</div></div>
    <div class="stat-card"><div class="value">{reg_stats['resid_std']}</div><div class="label">Residual Std (log)</div></div>
  </div>

  <div class="chart-card">{charts['regression']}</div>

  <h3>Forecasted Appreciation</h3>
  <div class="chart-card">{forecast_df.to_html(index=False, classes='data-table', border=0)}</div>

  {"<h3>Appreciation by Medium</h3><div class='chart-card'>" + medium_reg_df.to_html(index=False, classes='data-table', border=0) + "</div>" if len(medium_reg_df) > 0 else ""}
</section>

<section>
  <h2>Current Estimated Values</h2>
  <div class="insight">
    <strong>What would each lot be worth if sold today (2026)?</strong>
    Each lot's predicted value uses the hedonic model (controlling for medium and size),
    preserving its lot-specific premium/discount (residual) from the original sale.
    Estimates capped at 1.5x historical max (${sold['hammer_price_usd'].max() * 1.5:,.0f}).
  </div>

  <div class="chart-card">{charts['predicted_vs_actual']}</div>
  <div class="chart-card">{charts['gain_loss']}</div>

  <h3>Full Valuation Table</h3>
  <div class="chart-card">{val_table.to_html(index=False, classes='data-table', border=0)}</div>
</section>

<!-- ── Sale Type ──────────────────────────────────────────────────── -->
<section>
  <h2>Auction Type</h2>
  <div class="chart-card" style="max-width:500px;">{charts['sale_type']}</div>
</section>

</div>

<footer>
  F.N. Souza Auction Market Analysis | Data: Christie's ({stats['years_range']}) | Generated with Python + Plotly
</footer>

</body>
</html>"""

with open('souza_report.html', 'w') as f:
    f.write(html)

print("Report written to souza_report.html")
print(f"File size: {len(html) / 1024:.0f} KB")
