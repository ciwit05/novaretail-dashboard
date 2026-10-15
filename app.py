"""
NovaRetail Customer Intelligence Dashboard
==========================================
Interactive executive analytics dashboard built for Sophia Martinez,
Director of Customer Intelligence at NovaRetail.

Supports:
a) Identifying growth opportunities within the existing customer base
b) Detecting early warning signs of customer decline or reduced engagement
c) Recommending data-driven commercial and marketing actions
d) AI-powered customer intelligence querying via Google Gemini 2.5 Flash
"""

from datetime import datetime, date
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE & BRANDING CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NovaRetail Customer Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished executive appearance
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 500;
    }
    /* Action container styling */
    .recommendation-card {
        background-color: #ffffff;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Canonical color mapping for customer segments
SEGMENT_COLORS = {
    "Promising": "#10B981",   # Vibrant emerald green
    "Growth": "#3B82F6",      # Vibrant sapphire blue
    "Stable": "#F59E0B",      # Warm amber
    "Decline": "#EF4444",     # Coral red
    "Unlabeled": "#9CA3AF",   # Slate gray
}

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PREPARATION
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_path: str = "NR_dataset.xlsx") -> pd.DataFrame:
    """
    Load NovaRetail dataset using relative path and select only required variables.
    Excludes idx, TransactionID, and CustomerGender per business requirements.
    """
    cols_to_use = [
        "label",
        "CustomerID",
        "TransactionDate",
        "ProductCategory",
        "PurchaseAmount",
        "CustomerAgeGroup",
        "CustomerRegion",
        "CustomerSatisfaction",
        "RetailChannel",
    ]
    df = pd.read_excel(file_path, sheet_name="data", usecols=cols_to_use)
    return df


def prepare_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, map, and derive required analytical fields.
    """
    df = df_raw.copy()

    # 1. Fill missing label with 'Unlabeled'
    df["label"] = df["label"].fillna("Unlabeled").astype(str).str.strip()

    # 2. Map 35 messy ProductCategory values to 9 CategoryGroup buckets
    category_mapping = {
        "Electronics": ["Electronics", "Gaming"],
        "Clothing & Fashion": [
            "Clothing",
            "Fashion",
            "Fashion & Apparel",
            "Fashion Accessories",
            "Children's Clothing",
            "Sportswear",
        ],
        "Groceries": [
            "Groceries",
            "Grocery",
            "Grocery Items",
            "Food & Beverages",
        ],
        "Home & Furniture": [
            "Home Appliances",
            "Home & Garden",
            "Home Decor",
            "Home Improvement",
            "Furniture",
            "Furniture & Decor",
            "Gardening Tools",
        ],
        "Sports & Outdoors": [
            "Sporting Goods",
            "Sports Equipment",
            "Sports & Outdoors",
            "Outdoor Equipment",
        ],
        "Health & Beauty": [
            "Health & Wellness",
            "Health Supplements",
            "Health & Beauty",
            "Beauty Products",
            "Beauty & Personal Care",
            "Cosmetics",
        ],
        "Books": ["Books", "Books & Magazines"],
        "Toys": ["Toys", "Toys & Games"],
        "Other": ["Office Supplies", "Automotive"],
    }
    item_to_group = {
        item: group
        for group, items in category_mapping.items()
        for item in items
    }
    df["CategoryGroup"] = df["ProductCategory"].map(item_to_group).fillna("Other")

    # 3. Derived Fields
    # SatisfactionBand: Low (1-2), Neutral (3), High (4-5)
    def assign_satisfaction_band(score: int) -> str:
        if score <= 2:
            return "Low (1-2)"
        elif score == 3:
            return "Neutral (3)"
        else:
            return "High (4-5)"

    df["SatisfactionBand"] = df["CustomerSatisfaction"].apply(assign_satisfaction_band)

    # AtRisk: Decline segment OR satisfaction <= 2
    df["AtRisk"] = (df["label"] == "Decline") | (df["CustomerSatisfaction"] <= 2)

    # Week: Monday week-start date derived from TransactionDate
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["Week"] = (
        df["TransactionDate"]
        - pd.to_timedelta(df["TransactionDate"].dt.weekday, unit="D")
    ).dt.date

    return df


# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render sidebar filter controls with default 'all selected' and a reset button.
    Returns filtered DataFrame and handles empty selections gracefully.
    """
    st.sidebar.title("🎛️ Filter Panel")
    st.sidebar.markdown(
        "<small style='color: #64748b;'>Tailor views across segments, regions, categories, and channels.</small>",
        unsafe_allow_html=True,
    )

    all_segments = sorted(df["label"].unique().tolist())
    all_regions = sorted(df["CustomerRegion"].unique().tolist())
    all_categories = sorted(df["CategoryGroup"].unique().tolist())
    all_channels = sorted(df["RetailChannel"].unique().tolist())
    all_ages = sorted(df["CustomerAgeGroup"].unique().tolist())

    min_date = df["TransactionDate"].dt.date.min()
    max_date = df["TransactionDate"].dt.date.max()

    # Initialize session state keys for filters if not present
    if "filter_segment" not in st.session_state:
        st.session_state["filter_segment"] = all_segments
    if "filter_region" not in st.session_state:
        st.session_state["filter_region"] = all_regions
    if "filter_category" not in st.session_state:
        st.session_state["filter_category"] = all_categories
    if "filter_channel" not in st.session_state:
        st.session_state["filter_channel"] = all_channels
    if "filter_age" not in st.session_state:
        st.session_state["filter_age"] = all_ages
    if "filter_dates" not in st.session_state:
        st.session_state["filter_dates"] = (min_date, max_date)

    # Reset button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state["filter_segment"] = all_segments
        st.session_state["filter_region"] = all_regions
        st.session_state["filter_category"] = all_categories
        st.session_state["filter_channel"] = all_channels
        st.session_state["filter_age"] = all_ages
        st.session_state["filter_dates"] = (min_date, max_date)
        st.rerun()

    # Multiselect filter controls
    selected_segments = st.sidebar.multiselect(
        "Customer Segment (label)",
        options=all_segments,
        key="filter_segment",
    )
    selected_regions = st.sidebar.multiselect(
        "Customer Region",
        options=all_regions,
        key="filter_region",
    )
    selected_categories = st.sidebar.multiselect(
        "Product Category Group",
        options=all_categories,
        key="filter_category",
    )
    selected_channels = st.sidebar.multiselect(
        "Retail Channel",
        options=all_channels,
        key="filter_channel",
    )
    selected_ages = st.sidebar.multiselect(
        "Customer Age Group",
        options=all_ages,
        key="filter_age",
    )

    # Date range picker
    selected_dates = st.sidebar.date_input(
        "Transaction Date Range",
        min_value=min_date,
        max_value=max_date,
        key="filter_dates",
    )

    # Safe date range parsing
    if isinstance(selected_dates, (tuple, list)):
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
        elif len(selected_dates) == 1:
            start_date = end_date = selected_dates[0]
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = end_date = selected_dates

    # Filter execution
    df_filtered = df[
        (df["label"].isin(selected_segments))
        & (df["CustomerRegion"].isin(selected_regions))
        & (df["CategoryGroup"].isin(selected_categories))
        & (df["RetailChannel"].isin(selected_channels))
        & (df["CustomerAgeGroup"].isin(selected_ages))
        & (df["TransactionDate"].dt.date >= start_date)
        & (df["TransactionDate"].dt.date <= end_date)
    ]

    return df_filtered


# -----------------------------------------------------------------------------
# 4. KPI ROW COMPONENT
# -----------------------------------------------------------------------------
def render_kpi_row(df_filtered: pd.DataFrame):
    """
    Renders 6 executive metric cards for the active filter selection.
    Uses CustomerID.nunique() strictly for customer metrics.
    """
    total_rev = df_filtered["PurchaseAmount"].sum()
    avg_purchase = df_filtered["PurchaseAmount"].mean() if not df_filtered.empty else 0.0
    unique_custs = df_filtered["CustomerID"].nunique()
    avg_sat = df_filtered["CustomerSatisfaction"].mean() if not df_filtered.empty else 0.0

    decline_rev = df_filtered[df_filtered["label"] == "Decline"]["PurchaseAmount"].sum()
    pct_decline_rev = (decline_rev / total_rev * 100) if total_rev > 0 else 0.0

    at_risk_custs = df_filtered[df_filtered["AtRisk"]]["CustomerID"].nunique()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total Revenue", f"${total_rev:,.2f}")
    with c2:
        st.metric("Avg Purchase Amount", f"${avg_purchase:,.2f}")
    with c3:
        st.metric("Unique Customers", f"{unique_custs:,}")
    with c4:
        st.metric("Avg Satisfaction", f"{avg_sat:.2f} / 5.0")
    with c5:
        st.metric("Revenue from Decline", f"{pct_decline_rev:.1f}%")
    with c6:
        st.metric("At-Risk Customers", f"{at_risk_custs:,}")


# -----------------------------------------------------------------------------
# 5. TAB 1: OVERVIEW (Revenue Drivers)
# -----------------------------------------------------------------------------
def render_tab_overview(df_filtered: pd.DataFrame):
    """
    Tab 1: Identifies which customers and segments generate the most revenue.
    """
    st.markdown("### 📊 Revenue Generation & Channel Breakdown")
    st.markdown(
        "<small style='color: #64748b;'>Overview of total sales volume segmented by customer behavioral tier, geographic territory, category group, and retail channel.</small>",
        unsafe_allow_html=True,
    )

    # Exclude Unlabeled from segment-comparison charts
    df_seg_comp = df_filtered[df_filtered["label"] != "Unlabeled"]

    col1, col2 = st.columns(2)

    with col1:
        # 1. Revenue by Segment
        if not df_seg_comp.empty:
            seg_summary = (
                df_seg_comp.groupby("label")
                .agg(
                    revenue=("PurchaseAmount", "sum"),
                    transactions=("PurchaseAmount", "count"),
                )
                .reset_index()
            )
            fig_seg = px.bar(
                seg_summary,
                x="label",
                y="revenue",
                color="label",
                color_discrete_map=SEGMENT_COLORS,
                text=seg_summary.apply(
                    lambda r: f"${r.revenue:,.0f} (n={r.transactions})", axis=1
                ),
                title="Total Revenue by Customer Segment",
                labels={"revenue": "Total Revenue ($)", "label": "Segment"},
                template="plotly_white",
            )
            fig_seg.update_traces(textposition="outside")
            fig_seg.update_layout(showlegend=False, yaxis=dict(rangemode="tozero"))
            st.plotly_chart(fig_seg, use_container_width=True)
            st.caption("Sales distribution across behavioral segments (excluding unlabeled records).")
        else:
            st.info("No labeled segment records in current filter selection.")

        # 3. Revenue by Category Group
        cat_summary = (
            df_filtered.groupby("CategoryGroup")
            .agg(
                revenue=("PurchaseAmount", "sum"),
                transactions=("PurchaseAmount", "count"),
            )
            .reset_index()
            .sort_values("revenue", ascending=True)
        )
        fig_cat = px.bar(
            cat_summary,
            x="revenue",
            y="CategoryGroup",
            orientation="h",
            text=cat_summary.apply(
                lambda r: f"${r.revenue:,.0f} (n={r.transactions})", axis=1
            ),
            color_discrete_sequence=["#2563eb"],
            title="Total Revenue by Product Category Group",
            labels={"revenue": "Total Revenue ($)", "CategoryGroup": "Category Group"},
            template="plotly_white",
        )
        fig_cat.update_traces(textposition="outside")
        fig_cat.update_layout(xaxis=dict(rangemode="tozero"))
        st.plotly_chart(fig_cat, use_container_width=True)
        st.caption("Sales contribution of standardized merchandise groups across all channels.")

    with col2:
        # 2. Revenue by Region
        reg_summary = (
            df_filtered.groupby("CustomerRegion")
            .agg(
                revenue=("PurchaseAmount", "sum"),
                transactions=("PurchaseAmount", "count"),
            )
            .reset_index()
            .sort_values("revenue", ascending=False)
        )
        fig_reg = px.bar(
            reg_summary,
            x="CustomerRegion",
            y="revenue",
            color="CustomerRegion",
            color_discrete_sequence=px.colors.qualitative.Safe,
            text=reg_summary.apply(
                lambda r: f"${r.revenue:,.0f} (n={r.transactions})", axis=1
            ),
            title="Total Revenue by Customer Region",
            labels={"revenue": "Total Revenue ($)", "CustomerRegion": "Region"},
            template="plotly_white",
        )
        fig_reg.update_traces(textposition="outside")
        fig_reg.update_layout(showlegend=False, yaxis=dict(rangemode="tozero"))
        st.plotly_chart(fig_reg, use_container_width=True)
        st.caption("Omnichannel revenue generated across geographic customer regions.")

        # 4. Online vs Physical Revenue (Donut)
        chan_summary = (
            df_filtered.groupby("RetailChannel")
            .agg(
                revenue=("PurchaseAmount", "sum"),
                transactions=("PurchaseAmount", "count"),
            )
            .reset_index()
        )
        fig_donut = px.pie(
            chan_summary,
            names="RetailChannel",
            values="revenue",
            hole=0.5,
            color="RetailChannel",
            color_discrete_map={"Online": "#3b82f6", "Physical Store": "#10b981"},
            title="Revenue Share: Online vs. Physical Store",
            template="plotly_white",
        )
        fig_donut.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<extra></extra>",
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.caption("Channel revenue split comparing digital and brick-and-mortar storefronts.")

    # 5. Weekly Revenue Line Trend
    st.markdown("#### Weekly Revenue Trend (Short-Term Observation)")
    week_summary = (
        df_filtered.groupby("Week")
        .agg(
            revenue=("PurchaseAmount", "sum"),
            transactions=("PurchaseAmount", "count"),
        )
        .reset_index()
        .sort_values("Week")
    )
    fig_week = px.line(
        week_summary,
        x="Week",
        y="revenue",
        markers=True,
        text=week_summary.apply(
            lambda r: f"${r.revenue:,.0f} (n={r.transactions})", axis=1
        ),
        title="Weekly Revenue Flow (Week Start Date)",
        labels={"revenue": "Weekly Revenue ($)", "Week": "Week Starting"},
        template="plotly_white",
    )
    fig_week.update_traces(
        line_color="#4f46e5",
        line_width=3,
        marker=dict(size=8),
        textposition="top center",
    )
    st.plotly_chart(fig_week, use_container_width=True)
    st.caption("Note: Dataset spans approximately 5 weeks (2023-01-15 to 2023-02-20); data reflects short-term operational cadence rather than secular seasonality.")

    # 6. Top 10 Customers Table
    st.markdown("#### 🏆 Top 10 Customers by Revenue")
    st.caption("High-value accounts ranked by total spend within the filtered date and segment criteria.")

    top_cust = (
        df_filtered.groupby("CustomerID")
        .agg(
            Segment=("label", lambda s: s.mode().iloc[0] if not s.empty else "N/A"),
            Region=("CustomerRegion", lambda r: r.mode().iloc[0] if not r.empty else "N/A"),
            TotalRevenue=("PurchaseAmount", "sum"),
            Transactions=("PurchaseAmount", "count"),
            AvgSatisfaction=("CustomerSatisfaction", "mean"),
        )
        .reset_index()
        .sort_values("TotalRevenue", ascending=False)
        .head(10)
    )

    top_cust_display = top_cust.copy()
    top_cust_display["CustomerID"] = top_cust_display["CustomerID"].astype(str)
    top_cust_display["TotalRevenue"] = top_cust_display["TotalRevenue"].apply(lambda v: f"${v:,.2f}")
    top_cust_display["AvgSatisfaction"] = top_cust_display["AvgSatisfaction"].apply(lambda s: f"{s:.2f} / 5.0")
    top_cust_display.columns = [
        "Customer ID",
        "Primary Segment",
        "Primary Region",
        "Total Revenue",
        "Transactions (n)",
        "Avg Satisfaction",
    ]
    st.dataframe(top_cust_display, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 6. TAB 2: GROWTH OPPORTUNITIES (Goal A)
# -----------------------------------------------------------------------------
def render_tab_growth(df_filtered: pd.DataFrame):
    """
    Tab 2: Identifies growth opportunities within existing customer base.
    """
    st.markdown("### 🚀 Growth Opportunities & Expansion Pockets")
    st.markdown(
        "<small style='color: #64748b;'>Discover high-potential product categories, high-ticket segments, and regional opportunities across Promising and Growth customer tiers.</small>",
        unsafe_allow_html=True,
    )

    df_seg_comp = df_filtered[df_filtered["label"] != "Unlabeled"]
    pg_df = df_filtered[df_filtered["label"].isin(["Promising", "Growth"])]

    # Dynamic Insight Box for Top Growth Pocket
    if not pg_df.empty:
        pg_combos = (
            pg_df.groupby(["label", "CategoryGroup", "CustomerRegion"])
            .agg(
                total_revenue=("PurchaseAmount", "sum"),
                transactions=("PurchaseAmount", "count"),
                avg_ticket=("PurchaseAmount", "mean"),
            )
            .reset_index()
            .sort_values("total_revenue", ascending=False)
        )
        if not pg_combos.empty:
            top_pocket = pg_combos.iloc[0]
            st.info(
                f"💡 **Dynamic Growth Insight:** In this filtered sample, the **{top_pocket['label']}** segment "
                f"purchasing **{top_pocket['CategoryGroup']}** in the **{top_pocket['CustomerRegion']}** region "
                f"represents the top growth pocket, generating **${top_pocket['total_revenue']:,.2f}** "
                f"across **{int(top_pocket['transactions'])}** transactions (average ticket: **${top_pocket['avg_ticket']:,.2f}**)."
            )
    else:
        st.info("💡 Adjust filters to include Promising or Growth segments to compute dynamic growth pockets.")

    col1, col2 = st.columns(2)

    with col1:
        # 1. Heatmap: Revenue by Segment x Category Group
        if not df_seg_comp.empty:
            pivot_rev = df_seg_comp.pivot_table(
                index="CategoryGroup",
                columns="label",
                values="PurchaseAmount",
                aggfunc="sum",
                fill_value=0,
            )
            pivot_cnt = df_seg_comp.pivot_table(
                index="CategoryGroup",
                columns="label",
                values="PurchaseAmount",
                aggfunc="count",
                fill_value=0,
            )

            # Build cell annotations
            annotations = []
            for y_idx, cat in enumerate(pivot_rev.index):
                for x_idx, seg in enumerate(pivot_rev.columns):
                    val = pivot_rev.loc[cat, seg]
                    cnt = pivot_cnt.loc[cat, seg]
                    annotations.append(
                        dict(
                            x=seg,
                            y=cat,
                            text=f"${val:,.0f}<br><span style='font-size:10px;'>n={cnt}</span>",
                            showarrow=False,
                            font=dict(color="black" if val < pivot_rev.values.max() * 0.65 else "white", size=11),
                        )
                    )

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=pivot_rev.values,
                    x=pivot_rev.columns.tolist(),
                    y=pivot_rev.index.tolist(),
                    colorscale="Blues",
                    hoverongaps=False,
                    hovertemplate="<b>Category:</b> %{y}<br><b>Segment:</b> %{x}<br><b>Revenue:</b> $%{z:,.2f}<extra></extra>",
                )
            )
            fig_heat.update_layout(
                title="Revenue Matrix: Segment × Category Group",
                annotations=annotations,
                template="plotly_white",
                xaxis_title="Customer Segment",
                yaxis_title="Category Group",
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Cross-tabulation of total revenue and transaction volume (n) across segments and merchandise groups.")
        else:
            st.warning("No segment records available for heatmap.")

        # 3. Revenue from Promising & Growth by Region
        if not pg_df.empty:
            pg_reg = (
                pg_df.groupby(["CustomerRegion", "label"])
                .agg(
                    revenue=("PurchaseAmount", "sum"),
                    transactions=("PurchaseAmount", "count"),
                )
                .reset_index()
            )
            fig_pg_reg = px.bar(
                pg_reg,
                x="CustomerRegion",
                y="revenue",
                color="label",
                barmode="group",
                color_discrete_map=SEGMENT_COLORS,
                text=pg_reg.apply(
                    lambda r: f"${r.revenue:,.0f} (n={r.transactions})", axis=1
                ),
                title="Promising & Growth Revenue by Region",
                labels={"revenue": "Revenue ($)", "CustomerRegion": "Region", "label": "Segment"},
                template="plotly_white",
            )
            fig_pg_reg.update_traces(textposition="outside")
            fig_pg_reg.update_layout(yaxis=dict(rangemode="tozero"))
            st.plotly_chart(fig_pg_reg, use_container_width=True)
            st.caption("Comparison of expansion tiers (Promising vs. Growth) across regional sales territories.")
        else:
            st.info("No Promising or Growth records in active filter selection.")

    with col2:
        # 2. Average Purchase Amount by Segment (with n)
        if not df_seg_comp.empty:
            seg_avg = (
                df_seg_comp.groupby("label")
                .agg(
                    avg_ticket=("PurchaseAmount", "mean"),
                    transactions=("PurchaseAmount", "count"),
                )
                .reset_index()
            )
            fig_avg = px.bar(
                seg_avg,
                x="label",
                y="avg_ticket",
                color="label",
                color_discrete_map=SEGMENT_COLORS,
                text=seg_avg.apply(
                    lambda r: f"${r.avg_ticket:,.2f} (n={r.transactions})", axis=1
                ),
                title="Average Ticket Size by Segment (with Sample n)",
                labels={"avg_ticket": "Avg Purchase Amount ($)", "label": "Segment"},
                template="plotly_white",
            )
            fig_avg.update_traces(textposition="outside")
            fig_avg.update_layout(showlegend=False, yaxis=dict(rangemode="tozero"))
            st.plotly_chart(fig_avg, use_container_width=True)
            st.caption("Average basket transaction size by behavioral segment with sample counts.")
        else:
            st.warning("No segment records available for average purchase comparison.")

        # 4. Channel Mix by Segment (Stacked Bar)
        if not df_seg_comp.empty:
            chan_mix = (
                df_seg_comp.groupby(["label", "RetailChannel"])
                .agg(
                    revenue=("PurchaseAmount", "sum"),
                    transactions=("PurchaseAmount", "count"),
                )
                .reset_index()
            )
            fig_mix = px.bar(
                chan_mix,
                x="label",
                y="revenue",
                color="RetailChannel",
                color_discrete_map={"Online": "#3b82f6", "Physical Store": "#10b981"},
                barmode="stack",
                text=chan_mix.apply(lambda r: f"${r.revenue:,.0f}", axis=1),
                title="Channel Revenue Mix by Customer Segment",
                labels={"revenue": "Revenue ($)", "label": "Segment", "RetailChannel": "Channel"},
                template="plotly_white",
            )
            fig_mix.update_traces(textposition="inside")
            st.plotly_chart(fig_mix, use_container_width=True)
            st.caption("Omnichannel footprint showing the proportion of online vs. physical store revenue per segment.")


# -----------------------------------------------------------------------------
# 7. TAB 3: RISK & DECLINE SIGNALS (Goal B)
# -----------------------------------------------------------------------------
def render_tab_risk(df_filtered: pd.DataFrame):
    """
    Tab 3: Detects early warning signs of decline or reduced customer engagement.
    """
    st.markdown("### ⚠️ Risk Radar & Early Decline Indicators")
    st.markdown(
        "<small style='color: #64748b;'>Monitor customer friction, satisfaction drop-offs, and revenue vulnerability before customer churn materializes.</small>",
        unsafe_allow_html=True,
    )

    df_seg_comp = df_filtered[df_filtered["label"] != "Unlabeled"]
    decline_df = df_filtered[df_filtered["label"] == "Decline"]
    at_risk_df = df_filtered[df_filtered["AtRisk"]]

    # Dynamic Warning Box
    if not at_risk_df.empty:
        total_risk_rev = at_risk_df["PurchaseAmount"].sum()
        risk_cust_count = at_risk_df["CustomerID"].nunique()
        low_sat_count = (at_risk_df["CustomerSatisfaction"] <= 2).sum()
        
        # Identify top region for at risk
        top_risk_region = (
            at_risk_df["CustomerRegion"].value_counts().index[0]
            if not at_risk_df["CustomerRegion"].empty
            else "N/A"
        )
        top_risk_chan = (
            at_risk_df["RetailChannel"].value_counts().index[0]
            if not at_risk_df["RetailChannel"].empty
            else "N/A"
        )

        st.warning(
            f"🚨 **Risk Summary Alert:** **{risk_cust_count}** unique customer(s) are currently flagged as At-Risk "
            f"(accounting for **${total_risk_rev:,.2f}** in exposed revenue across **{len(at_risk_df)}** transactions). "
            f"**{low_sat_count}** orders have severe dissatisfaction ratings (1-2 / 5.0). "
            f"The highest density of at-risk transactions is concentrated in the **{top_risk_region}** region "
            f"via the **{top_risk_chan}** channel."
        )
    else:
        st.success("✅ No at-risk customers or low-satisfaction signals in the active filter selection.")

    col1, col2 = st.columns(2)

    with col1:
        # 1. Satisfaction by Segment
        if not df_seg_comp.empty:
            sat_summary = (
                df_seg_comp.groupby("label")
                .agg(
                    avg_sat=("CustomerSatisfaction", "mean"),
                    transactions=("CustomerSatisfaction", "count"),
                )
                .reset_index()
            )
            fig_sat = px.bar(
                sat_summary,
                x="label",
                y="avg_sat",
                color="label",
                color_discrete_map=SEGMENT_COLORS,
                text=sat_summary.apply(
                    lambda r: f"{r.avg_sat:.2f} / 5.0 (n={r.transactions})", axis=1
                ),
                title="Average Customer Satisfaction by Segment",
                labels={"avg_sat": "Avg Satisfaction (1-5)", "label": "Segment"},
                template="plotly_white",
            )
            fig_sat.update_traces(textposition="outside")
            fig_sat.update_layout(showlegend=False, yaxis=dict(range=[0, 5.5]))
            st.plotly_chart(fig_sat, use_container_width=True)
            st.caption("Satisfaction score comparison across segments; notice how Decline correlates with lower satisfaction.")
        else:
            st.info("No segment records available for satisfaction comparison.")

        # 3. Decline Transactions by Region and Channel
        if not decline_df.empty:
            dec_reg = (
                decline_df.groupby(["CustomerRegion", "RetailChannel"])
                .agg(
                    revenue=("PurchaseAmount", "sum"),
                    transactions=("PurchaseAmount", "count"),
                )
                .reset_index()
            )
            fig_dec = px.bar(
                dec_reg,
                x="CustomerRegion",
                y="transactions",
                color="RetailChannel",
                barmode="group",
                color_discrete_map={"Online": "#ef4444", "Physical Store": "#f97316"},
                text="transactions",
                title="Decline Customer Transactions by Region & Channel",
                labels={"transactions": "Decline Transactions (n)", "CustomerRegion": "Region"},
                template="plotly_white",
            )
            fig_dec.update_traces(textposition="outside")
            fig_dec.update_layout(yaxis=dict(rangemode="tozero"))
            st.plotly_chart(fig_dec, use_container_width=True)
            st.caption("Count of Decline customer orders broken down by customer region and fulfillment channel.")
        else:
            st.info("No Decline segment transactions found in the active filter selection.")

    with col2:
        # 2. Distribution of Satisfaction Bands
        band_order = ["Low (1-2)", "Neutral (3)", "High (4-5)"]
        band_summary = (
            df_filtered.groupby("SatisfactionBand")
            .agg(
                transactions=("PurchaseAmount", "count"),
                revenue=("PurchaseAmount", "sum"),
            )
            .reindex(band_order)
            .fillna(0)
            .reset_index()
        )
        fig_band = px.bar(
            band_summary,
            x="SatisfactionBand",
            y="transactions",
            color="SatisfactionBand",
            color_discrete_map={
                "Low (1-2)": "#ef4444",
                "Neutral (3)": "#f59e0b",
                "High (4-5)": "#10b981",
            },
            text=band_summary.apply(
                lambda r: f"{int(r.transactions)} orders (${r.revenue:,.0f})", axis=1
            ),
            title="Distribution of Customer Satisfaction Bands",
            labels={"transactions": "Number of Orders", "SatisfactionBand": "Satisfaction Band"},
            template="plotly_white",
        )
        fig_band.update_traces(textposition="outside")
        fig_band.update_layout(showlegend=False, yaxis=dict(rangemode="tozero"))
        st.plotly_chart(fig_band, use_container_width=True)
        st.caption("Order volume across satisfaction tiers (Low: scores 1-2, Neutral: score 3, High: scores 4-5).")

        # 4. Average Purchase Amount: Decline vs. Other Segments
        if not df_seg_comp.empty:
            df_comp = df_seg_comp.copy()
            df_comp["SegmentCohort"] = df_comp["label"].apply(
                lambda x: "Decline Segment" if x == "Decline" else "Other Segments (Promising/Growth/Stable)"
            )
            cohort_summary = (
                df_comp.groupby("SegmentCohort")
                .agg(
                    avg_ticket=("PurchaseAmount", "mean"),
                    transactions=("PurchaseAmount", "count"),
                )
                .reset_index()
            )
            fig_cohort = px.bar(
                cohort_summary,
                x="SegmentCohort",
                y="avg_ticket",
                color="SegmentCohort",
                color_discrete_map={
                    "Decline Segment": "#ef4444",
                    "Other Segments (Promising/Growth/Stable)": "#3b82f6",
                },
                text=cohort_summary.apply(
                    lambda r: f"${r.avg_ticket:,.2f} (n={r.transactions})", axis=1
                ),
                title="Average Spend: Decline vs. Non-Decline Segments",
                labels={"avg_ticket": "Avg Purchase Amount ($)", "SegmentCohort": "Cohort"},
                template="plotly_white",
            )
            fig_cohort.update_traces(textposition="outside")
            fig_cohort.update_layout(showlegend=False, yaxis=dict(rangemode="tozero"))
            st.plotly_chart(fig_cohort, use_container_width=True)
            st.caption("Evaluation of ticket size difference between declining customers and engaged segments.")
        else:
            st.info("No segment records available for cohort ticket comparison.")

    # 5. At-Risk Customers Detail Table
    st.markdown("#### 🔍 Filtered At-Risk Customer Directory")
    st.caption("Customers flagged due to 'Decline' status or low satisfaction (<= 2). Prioritized by lowest satisfaction.")

    if not at_risk_df.empty:
        at_risk_display = (
            at_risk_df[
                [
                    "CustomerID",
                    "label",
                    "CustomerRegion",
                    "RetailChannel",
                    "CategoryGroup",
                    "CustomerSatisfaction",
                    "PurchaseAmount",
                ]
            ]
            .sort_values(
                by=["CustomerSatisfaction", "PurchaseAmount"],
                ascending=[True, False],
            )
            .copy()
        )
        at_risk_display["CustomerID"] = at_risk_display["CustomerID"].astype(str)
        at_risk_display["PurchaseAmount"] = at_risk_display["PurchaseAmount"].apply(lambda v: f"${v:,.2f}")
        at_risk_display.columns = [
            "Customer ID",
            "Segment",
            "Region",
            "Channel",
            "Category Group",
            "Satisfaction Rating",
            "Purchase Amount",
        ]
        st.dataframe(at_risk_display, use_container_width=True, hide_index=True)
    else:
        st.info("No at-risk customer records match the current filter selection.")


# -----------------------------------------------------------------------------
# 8. TAB 4: RECOMMENDATIONS (Goal C)
# -----------------------------------------------------------------------------
def render_tab_recommendations(df_filtered: pd.DataFrame):
    """
    Tab 4: Recommends commercial and marketing actions backed by filtered metrics.
    """
    st.markdown("### 📋 Executive Action Cards & Prioritization Matrix")
    st.markdown(
        "<small style='color: #64748b;'>Concrete, data-backed operational and marketing interventions tailored for Sophia Martinez and the retail leadership team.</small>",
        unsafe_allow_html=True,
    )

    # Compute key figures for recommendations
    total_rev = df_filtered["PurchaseAmount"].sum()
    n_records = len(df_filtered)

    pg_df = df_filtered[df_filtered["label"].isin(["Promising", "Growth"])]
    pg_rev = pg_df["PurchaseAmount"].sum()
    pg_custs = pg_df["CustomerID"].nunique()
    top_pg_cat = (
        pg_df.groupby("CategoryGroup")["PurchaseAmount"].sum().idxmax()
        if not pg_df.empty
        else "Key Merchandise"
    )

    at_risk_df = df_filtered[df_filtered["AtRisk"]]
    at_risk_rev = at_risk_df["PurchaseAmount"].sum()
    at_risk_custs = at_risk_df["CustomerID"].nunique()

    stable_df = df_filtered[df_filtered["label"] == "Stable"]
    stable_rev = stable_df["PurchaseAmount"].sum()
    stable_custs = stable_df["CustomerID"].nunique()
    stable_avg = stable_df["PurchaseAmount"].mean() if not stable_df.empty else 0.0

    online_df = df_filtered[df_filtered["RetailChannel"] == "Online"]
    store_df = df_filtered[df_filtered["RetailChannel"] == "Physical Store"]
    online_sat = online_df["CustomerSatisfaction"].mean() if not online_df.empty else 0.0
    store_sat = store_df["CustomerSatisfaction"].mean() if not store_df.empty else 0.0

    st.markdown("#### 🎯 Strategic Action Cards")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### 1. Invest in High-Momentum Segments")
            st.markdown(
                f"**Focus Area:** Category expansion in **{top_pg_cat}** for Promising & Growth tiers.<br>"
                f"**Supporting Evidence:** In this filtered sample of {n_records} transactions, Promising and Growth "
                f"customers generated **${pg_rev:,.2f}** ({pg_rev / total_rev * 100:.1f}% of filtered revenue) across "
                f"**{pg_custs}** unique customer accounts.<br>"
                f"**Recommended Action:** Allocate VIP loyalty incentives, early-access marketing promos, and category bundling "
                f"around {top_pg_cat} to accelerate customer migration into top spenders.",
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("#### 2. Retain High-Value At-Risk Customers")
            st.markdown(
                f"**Focus Area:** Proactive intervention for accounts with declining spend or satisfaction ≤ 2.<br>"
                f"**Supporting Evidence:** In this filtered sample, **{at_risk_custs}** accounts are flagged At-Risk, "
                f"accounting for **${at_risk_rev:,.2f}** ({at_risk_rev / total_rev * 100:.1f}% of filtered revenue).<br>"
                f"**Recommended Action:** Deploy automated customer-care triggers: executive apology emails, "
                f"immediate 20% discount vouchers, and direct CS outreach within 24 hours of low rating submission.",
                unsafe_allow_html=True,
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### 3. Re-engage Touchpoints with Satisfaction Gaps")
            st.markdown(
                f"**Focus Area:** Friction resolution across retail touchpoints.<br>"
                f"**Supporting Evidence:** In this sample, Online channel satisfaction averages **{online_sat:.2f} / 5.0**, "
                f"compared to **{store_sat:.2f} / 5.0** in Physical Stores.<br>"
                f"**Recommended Action:** Conduct operational audit of checkout friction, in-store staff training, and "
                f"online returns/shipping reliability to close the satisfaction deficit.",
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("#### 4. Upsell & Stabilize Base Customers")
            st.markdown(
                f"**Focus Area:** Frequency acceleration for Stable customer segment.<br>"
                f"**Supporting Evidence:** In this sample, Stable customers represent **${stable_rev:,.2f}** across "
                f"**{stable_custs}** customers with an average order value of **${stable_avg:,.2f}**.<br>"
                f"**Recommended Action:** Launch subscription/replenishment reminders and omnichannel crossover promotions "
                f"(e.g., 'Buy online, pick up in store for a $15 store credit').",
                unsafe_allow_html=True,
            )

    st.markdown("#### 🧭 Decision-Maker Priority Matrix")
    priority_data = [
        {
            "Segment": "Decline & Low Satisfaction",
            "Recommended Action": "High-priority outreach & CS resolution protocol to prevent permanent customer attrition",
            "Supporting Metric": f"In this sample: {at_risk_custs} at-risk customers, ${at_risk_rev:,.2f} revenue exposed",
            "Priority": "High",
        },
        {
            "Segment": "Promising & Growth",
            "Recommended Action": f"Increase marketing spend on high-performing {top_pg_cat} & personalized upsell bundles",
            "Supporting Metric": f"In this sample: ${pg_rev:,.2f} revenue across {pg_custs} accounts",
            "Priority": "High",
        },
        {
            "Segment": "Stable",
            "Recommended Action": "Drive omnichannel cross-shopping incentives to increase basket size and visit frequency",
            "Supporting Metric": f"In this sample: ${stable_rev:,.2f} total revenue, avg ticket ${stable_avg:,.2f}",
            "Priority": "Medium",
        },
        {
            "Segment": "Unlabeled",
            "Recommended Action": "Enrich transactional and demographic tracking to ensure 100% behavioral classification",
            "Supporting Metric": f"In this sample: {df_filtered[df_filtered['label'] == 'Unlabeled']['CustomerID'].nunique()} unclassified customer(s)",
            "Priority": "Low",
        },
    ]
    st.dataframe(pd.DataFrame(priority_data), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 9. TAB 5: AI CHATBOT (Google Gemini 2.5 Flash)
# -----------------------------------------------------------------------------
def build_context(df_filtered: pd.DataFrame) -> str:
    """
    Build a rich, structured text summary of the CURRENTLY FILTERED dataset
    to inject into the Gemini context.
    """
    total_rev = df_filtered["PurchaseAmount"].sum()
    unique_custs = df_filtered["CustomerID"].nunique()
    total_trans = len(df_filtered)
    avg_sat = df_filtered["CustomerSatisfaction"].mean() if not df_filtered.empty else 0.0

    # Segment metrics
    seg_stats = (
        df_filtered.groupby("label")
        .agg(
            revenue=("PurchaseAmount", "sum"),
            transactions=("PurchaseAmount", "count"),
            customers=("CustomerID", "nunique"),
            avg_satisfaction=("CustomerSatisfaction", "mean"),
        )
        .to_dict(orient="index")
    )

    # Region metrics
    reg_stats = (
        df_filtered.groupby("CustomerRegion")
        .agg(
            revenue=("PurchaseAmount", "sum"),
            transactions=("PurchaseAmount", "count"),
            customers=("CustomerID", "nunique"),
        )
        .to_dict(orient="index")
    )

    # Category metrics
    cat_stats = (
        df_filtered.groupby("CategoryGroup")
        .agg(
            revenue=("PurchaseAmount", "sum"),
            transactions=("PurchaseAmount", "count"),
        )
        .to_dict(orient="index")
    )

    # Channel metrics
    chan_stats = (
        df_filtered.groupby("RetailChannel")
        .agg(
            revenue=("PurchaseAmount", "sum"),
            transactions=("PurchaseAmount", "count"),
            avg_satisfaction=("CustomerSatisfaction", "mean"),
        )
        .to_dict(orient="index")
    )

    # At-risk list
    at_risk = df_filtered[df_filtered["AtRisk"]]
    at_risk_list = []
    for _, row in at_risk.head(15).iterrows():
        at_risk_list.append(
            f"CustomerID: {row['CustomerID']}, Segment: {row['label']}, Region: {row['CustomerRegion']}, "
            f"Channel: {row['RetailChannel']}, Category: {row['CategoryGroup']}, Sat: {row['CustomerSatisfaction']}, "
            f"Spend: ${row['PurchaseAmount']:.2f}"
        )

    context_str = f"""
### CURRENT ACTIVE DASHBOARD CONTEXT:
- Total Filtered Revenue: ${total_rev:,.2f}
- Unique Customers in View: {unique_custs}
- Total Transactions in View: {total_trans}
- Average Satisfaction Rating: {avg_sat:.2f} / 5.0
- Date Range in View: {df_filtered['TransactionDate'].dt.date.min()} to {df_filtered['TransactionDate'].dt.date.max()}

### BY CUSTOMER SEGMENT:
{seg_stats}

### BY CUSTOMER REGION:
{reg_stats}

### BY PRODUCT CATEGORY GROUP:
{cat_stats}

### BY RETAIL CHANNEL:
{chan_stats}

### AT-RISK CUSTOMER SAMPLE (Top {len(at_risk_list)} flagged records):
{chr(10).join(at_risk_list) if at_risk_list else "None"}
"""
    return context_str


def render_tab_chatbot(df_filtered: pd.DataFrame):
    """
    Tab 5: AI Customer Intelligence Chatbot powered by Google Gemini 2.5 Flash.
    Reads GEMINI_API_KEY from st.secrets without hard-coding.
    """
    st.markdown("### 🤖 NovaRetail AI Customer Intelligence Advisor")
    st.markdown(
        "<small style='color: #64748b;'>Ask strategic questions directly to the Gemini 2.5 Flash model, grounded strictly in your current filtered dataset.</small>",
        unsafe_allow_html=True,
    )

    # Retrieve API key strictly from st.secrets with safe fallback
    gemini_api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
        elif "gemini_api_key" in st.secrets:
            gemini_api_key = st.secrets["gemini_api_key"]
    except Exception:
        gemini_api_key = None

    if not gemini_api_key:
        st.info(
            "ℹ️ **Gemini API Key Required**: To chat with the AI Customer Intelligence Assistant, "
            "please configure `GEMINI_API_KEY` in `.streamlit/secrets.toml` (locally) or in the Streamlit Community Cloud "
            "Secrets manager. All core dashboard analytics and visualizations remain 100% active."
        )

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello Sophia! I am your NovaRetail Customer Intelligence Analyst. "
                    "I am actively synchronized with your filtered dashboard view. "
                    "Ask me about top revenue drivers, at-risk customer patterns, or growth opportunities!"
                ),
            }
        ]

    # Clickable Suggested Questions
    st.markdown("##### 💡 Suggested Questions")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    suggested_prompt = None

    with s_col1:
        if st.button("📈 Which segment drives the most revenue?", use_container_width=True):
            suggested_prompt = "Which segment generates the most revenue, and what are their key characteristics?"
    with s_col2:
        if st.button("⚠️ Where are the biggest signs of decline?", use_container_width=True):
            suggested_prompt = "Where are the biggest signs of customer decline in this data, and what channels or regions are most exposed?"
    with s_col3:
        if st.button("🎯 Which category should we invest in?", use_container_width=True):
            suggested_prompt = "Which product category group should we invest in for maximum customer growth?"
    with s_col4:
        if st.button("🏬 How do Online & Stores compare?", use_container_width=True):
            suggested_prompt = "How do online and physical store channels compare in terms of revenue and customer satisfaction?"

    # Render previous messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input handling (via chat input or suggested button)
    user_prompt = st.chat_input("Ask a question about the customer intelligence data...")
    if suggested_prompt:
        user_prompt = suggested_prompt

    if user_prompt:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate response using Google Gemini
        if not gemini_api_key:
            reply = (
                "⚠️ **API Key Missing**: The Gemini API key has not been configured in `st.secrets['GEMINI_API_KEY']`. "
                "Please add it to your `.streamlit/secrets.toml` file to activate AI querying."
            )
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing customer intelligence data..."):
                    try:
                        from google import genai
                        from google.genai import types

                        client = genai.Client(api_key=gemini_api_key)

                        dataset_context = build_context(df_filtered)

                        system_instruction = (
                            "You are the senior Customer Intelligence AI Advisor for Sophia Martinez, "
                            "Director of Customer Intelligence at NovaRetail. "
                            "Answer questions based STRICTLY on the filtered dataset context provided below.\n\n"
                            "GUIDELINES:\n"
                            "1. Answer ONLY using the facts and metrics provided in the context.\n"
                            "2. Always cite specific numbers (revenue in $, transaction counts n, satisfaction scores).\n"
                            "3. Mention small sample sizes when relevant (note that the total dataset is only 100 rows "
                            "and spans approximately 5 weeks from Jan 15 to Feb 20, 2023).\n"
                            "4. If the provided context does not have enough data to answer the user's question, "
                            "explicitly state so rather than hallucinating or speculating.\n"
                            "5. Structure answers with clear bullet points, bold key figures, and executive brevity."
                        )

                        full_prompt = f"{dataset_context}\n\nUSER QUESTION: {user_prompt}"

                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=full_prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=0.2,
                            ),
                        )
                        reply = response.text
                        st.markdown(reply)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": reply}
                        )

                    except Exception as e:
                        err_msg = (
                            f"⚠️ **Error connecting to Google Gemini**: {str(e)}\n\n"
                            "Please check your API key, network connection, or rate limits."
                        )
                        st.error(err_msg)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": err_msg}
                        )


# -----------------------------------------------------------------------------
# 10. MAIN APPLICATION CONTROLLER
# -----------------------------------------------------------------------------
def main():
    # 1. Load Data
    try:
        df_raw = load_data("NR_dataset.xlsx")
    except Exception as e:
        st.error(
            f"❌ **Fatal Error loading dataset**: Could not find or read 'NR_dataset.xlsx' from current folder. Error: {e}"
        )
        st.stop()

    # 2. Prepare Data
    df = prepare_data(df_raw)

    # 3. Sidebar Filters
    df_filtered = apply_filters(df)

    # 4. Header & Subtitle
    st.title("🛍️ NovaRetail Customer Intelligence Dashboard")
    st.markdown(
        f"**Executive Intelligence Dashboard for Sophia Martinez, Director of Customer Intelligence**  \n"
        f"Active Filter View: **{df_filtered['CustomerID'].nunique()}** unique customers | **{len(df_filtered)}** transactions "
        f"(out of {df['CustomerID'].nunique()} total customers and {len(df)} transactions)"
    )
    st.markdown("---")

    # 5. Empty Data Guard
    if df_filtered.empty:
        st.warning(
            "⚠️ **No transactions match the selected filter criteria.**  \n"
            "Please adjust or reset your filters in the sidebar panel to restore data visualization."
        )
        st.stop()

    # 6. KPI Row
    render_kpi_row(df_filtered)
    st.markdown("<br>", unsafe_allow_html=True)

    # 7. Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Overview",
            "🚀 Growth Opportunities",
            "⚠️ Risk & Decline Signals",
            "📋 Recommendations",
            "🤖 AI Chatbot",
        ]
    )

    with tab1:
        render_tab_overview(df_filtered)

    with tab2:
        render_tab_growth(df_filtered)

    with tab3:
        render_tab_risk(df_filtered)

    with tab4:
        render_tab_recommendations(df_filtered)

    with tab5:
        render_tab_chatbot(df_filtered)


if __name__ == "__main__":
    main()
