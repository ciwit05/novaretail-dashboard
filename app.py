"""
NovaRetail Customer Intelligence Dashboard
==========================================
Interactive executive analytics dashboard built for Sophia Martinez,
Director of Customer Intelligence at NovaRetail.

Supports:
a) Identifying growth opportunities within the existing customer base
b) Detecting early warning signs of customer decline or reduced engagement
c) Recommending data-driven commercial and marketing actions
d) AI-powered customer intelligence querying via Groq (Llama 3.3 70B Versatile)
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
    </style>
    """,
    unsafe_allow_html=True,
)


def escape_dollars(text: str) -> str:
    """
    Helper that replaces '$' with '\\$' so that Streamlit markdown parser
    does not mistakenly parse dollar amounts as LaTeX math delimiters.
    """
    if text is None:
        return ""
    return str(text).replace("$", r"\$")

# Canonical color mapping for customer segments
SEGMENT_COLORS = {
    "Promising": "#10B981",   # Vibrant emerald green
    "Growth": "#3B82F6",      # Vibrant sapphire blue
    "Stable": "#F59E0B",      # Warm amber
    "Decline": "#EF4444",     # Coral red
    "Unlabeled": "#9CA3AF",   # Slate gray
}

# Channel color mapping (Item 16: Online = teal, Physical Store = purple)
CHANNEL_COLORS = {
    "Online": "#0d9488",          # Modern teal
    "Physical Store": "#7c3aed",  # Modern purple
}

# Neutral palette for geographic regions (Item 16)
NEUTRAL_REGION_COLORS = ["#334155", "#475569", "#64748b", "#94a3b8"]


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
    if "filter_start_date" not in st.session_state:
        st.session_state["filter_start_date"] = min_date
    if "filter_end_date" not in st.session_state:
        st.session_state["filter_end_date"] = max_date

    # Reset button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state["filter_segment"] = all_segments
        st.session_state["filter_region"] = all_regions
        st.session_state["filter_category"] = all_categories
        st.session_state["filter_channel"] = all_channels
        st.session_state["filter_age"] = all_ages
        st.session_state["filter_start_date"] = min_date
        st.session_state["filter_end_date"] = max_date
        st.rerun()

    # Multiselect filter controls
    selected_segments = st.sidebar.multiselect(
        "Segment label (applies to transactions)",
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

    # Date range picker (Item 18: two date_inputs in columns for full visibility)
    st.sidebar.markdown("**Transaction Date Range**")
    d_col1, d_col2 = st.sidebar.columns(2)
    with d_col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.get("filter_start_date", min_date),
            min_value=min_date,
            max_value=max_date,
            key="filter_start_date",
        )
    with d_col2:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.get("filter_end_date", max_date),
            min_value=min_date,
            max_value=max_date,
            key="filter_end_date",
        )

    # Ensure start_date does not exceed end_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date

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

    at_risk_txns = int(df_filtered["AtRisk"].sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total Revenue", f"${total_rev:,.2f}")
    with c2:
        st.metric("Avg Purchase Amount", f"${avg_purchase:,.2f}")
    with c3:
        st.metric("Unique Customer IDs", f"{unique_custs:,}")
    with c4:
        st.metric("Avg Satisfaction", f"{avg_sat:.2f} / 5.0")
    with c5:
        st.metric("Revenue from Decline Txns", f"{pct_decline_rev:.1f}%")
    with c6:
        st.metric("At-Risk Transactions", f"{at_risk_txns:,}")


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

        # 3. Revenue by Category Group (Item 14: cliponaxis=False and extend range 15% beyond max)
        cat_summary = (
            df_filtered.groupby("CategoryGroup")
            .agg(
                revenue=("PurchaseAmount", "sum"),
                transactions=("PurchaseAmount", "count"),
            )
            .reset_index()
            .sort_values("revenue", ascending=True)
        )
        max_cat_rev = cat_summary["revenue"].max() if not cat_summary.empty else 0
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
        fig_cat.update_traces(textposition="outside", cliponaxis=False)
        fig_cat.update_layout(
            xaxis=dict(
                rangemode="tozero",
                range=[0, max_cat_rev * 1.15 if max_cat_rev > 0 else 100],
            )
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        st.caption("Sales contribution of standardized merchandise groups across all channels.")

    with col2:
        # 2. Revenue by Region (Item 16: neutral palette for regions)
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
            color_discrete_sequence=NEUTRAL_REGION_COLORS,
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

        # 4. Online vs Physical Revenue (Donut) (Item 15 & 16: percent only, inside, horizontal, side legend, teal/purple)
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
            color_discrete_map=CHANNEL_COLORS,
            title="Revenue Share: Online vs. Physical Store",
            template="plotly_white",
        )
        fig_donut.update_traces(
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<extra></extra>",
        )
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
            ),
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
    st.markdown("#### 🏆 Top 10 Customer IDs by Revenue")
    st.caption("Customer IDs ranked by total spend in the filtered view. Segment labels apply to transactions, so one ID can carry several labels (see 'Segments seen').")

    top_cust = (
        df_filtered.groupby("CustomerID")
        .agg(
            Segment=("label", lambda s: s.mode().iloc[0] if not s.empty else "N/A"),
            Region=("CustomerRegion", lambda r: r.mode().iloc[0] if not r.empty else "N/A"),
            TotalRevenue=("PurchaseAmount", "sum"),
            Transactions=("PurchaseAmount", "count"),
            AvgSatisfaction=("CustomerSatisfaction", "mean"),
            SegmentsSeen=("label", lambda s: ", ".join(sorted(s.unique()))),
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
        "Most Frequent Segment",
        "Most Frequent Region",
        "Total Revenue",
        "Transactions (n)",
        "Avg Satisfaction",
        "Segments seen",
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
            qualifying_combos = pg_combos[pg_combos["transactions"] >= 3].sort_values(
                "total_revenue", ascending=False
            )
            if not qualifying_combos.empty:
                top_pocket = qualifying_combos.iloc[0]
                st.info(
                    escape_dollars(
                        f"💡 **Dynamic Growth Insight:** In this filtered sample (requiring n ≥ 3), the **{top_pocket['label']}** segment "
                        f"purchasing **{top_pocket['CategoryGroup']}** in the **{top_pocket['CustomerRegion']}** region "
                        f"represents the top growth pocket, generating **${top_pocket['total_revenue']:,.2f}** "
                        f"across **n = {int(top_pocket['transactions'])}** transactions (average ticket: **${top_pocket['avg_ticket']:,.2f}**)."
                    )
                )
            else:
                st.info(
                    "💡 **Dynamic Growth Insight:** There is not enough data: No segment/category/region combination in "
                    "Promising or Growth meets the minimum threshold of at least 3 transactions (n ≥ 3) in the active filter selection."
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

            # Build cell annotations (Item 13: one non-overlapping label "$X (n=Y)", smaller font, no labels on empty cells)
            annotations = []
            for y_idx, cat in enumerate(pivot_rev.index):
                for x_idx, seg in enumerate(pivot_rev.columns):
                    val = pivot_rev.loc[cat, seg]
                    cnt = pivot_cnt.loc[cat, seg]
                    if cnt > 0 and val > 0:
                        annotations.append(
                            dict(
                                x=seg,
                                y=cat,
                                text=f"${val:,.0f} (n={cnt})",
                                showarrow=False,
                                font=dict(
                                    color="black" if val < pivot_rev.values.max() * 0.65 else "white",
                                    size=9.5,
                                ),
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
                height=520,
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

        # 4. Channel Mix by Segment (Stacked Bar) (Item 16: Online=teal, Physical Store=purple)
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
                color_discrete_map=CHANNEL_COLORS,
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
            escape_dollars(
                f"🚨 **Risk Summary Alert:** **{len(at_risk_df)}** transaction(s) across **{risk_cust_count}** customer ID(s) are currently flagged as At-Risk "
                f"(accounting for **${total_risk_rev:,.2f}** in exposed revenue). "
                f"**{low_sat_count}** orders have severe dissatisfaction ratings (1-2 / 5.0). "
                f"The highest density of at-risk transactions is concentrated in the **{top_risk_region}** region "
                f"via the **{top_risk_chan}** channel."
            )
        )
    else:
        st.success("✅ No at-risk transactions or low-satisfaction signals in the active filter selection.")

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
                color_discrete_map=CHANNEL_COLORS,
                text="transactions",
                title="Decline-Labeled Transactions by Region & Channel",
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

        # 4. Average Purchase Amount: Decline vs. Other Segments (Item 16: gray for Other segments)
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
                    "Other Segments (Promising/Growth/Stable)": "#9ca3af",
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
            st.caption("Average ticket of Decline-labeled transactions vs. all other labeled transactions.")
        else:
            st.info("No segment records available for cohort ticket comparison.")

    # 5. At-Risk Customers Detail Table
    st.markdown("#### 🔍 Filtered At-Risk Transaction Directory")
    st.caption("Transactions flagged because they are labeled 'Decline' or have satisfaction <= 2, sorted by lowest satisfaction. A customer ID can appear in more than one segment.")

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
        st.info("No at-risk transactions match the current filter selection.")

    # 6. Watchlist (Early Warning) Section (Item 9)
    st.markdown("---")
    st.markdown("#### 👁️ Watchlist (Early Warning Signals)")
    st.caption("Transactions NOT in the Decline segment with customer satisfaction ≤ 3 (neutral or low). Early detection helps prevent customer attrition.")

    watchlist_df = df_filtered[
        (df_filtered["label"] != "Decline") & (df_filtered["CustomerSatisfaction"] <= 3)
    ]
    w_cnt = len(watchlist_df)
    w_custs = watchlist_df["CustomerID"].nunique()
    w_rev = watchlist_df["PurchaseAmount"].sum()

    st.markdown(
        escape_dollars(
            f"**Watchlist Volume:** **{w_cnt}** transaction(s) across **{w_custs}** customer ID(s) totaling **${w_rev:,.2f}** in revenue."
        )
    )

    if not watchlist_df.empty:
        st.info(
            escape_dollars(
                f"ℹ️ **Early Warning Signal:** In this filtered sample, **{w_cnt}** transaction(s) across **{w_custs}** customer ID(s) "
                f"outside the Decline segment report satisfaction ratings ≤ 3, accounting for **${w_rev:,.2f}** in active spend. "
                "Proactively addressing friction points in these transactions can help prevent downward migration into the Decline segment."
            )
        )
        watchlist_display = (
            watchlist_df[
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
        watchlist_display["CustomerID"] = watchlist_display["CustomerID"].astype(str)
        watchlist_display["PurchaseAmount"] = watchlist_display["PurchaseAmount"].apply(lambda v: f"${v:,.2f}")
        watchlist_display.columns = [
            "Customer ID",
            "Segment",
            "Region",
            "Channel",
            "Category Group",
            "Satisfaction Rating",
            "Purchase Amount",
        ]
        st.dataframe(watchlist_display, use_container_width=True, hide_index=True)
    else:
        st.info("✅ No non-Decline transactions with satisfaction ≤ 3 in the active filter selection.")


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

    # Top 5 at-risk transactions by purchase amount (Item 11a)
    top_at_risk_list = []
    if not at_risk_df.empty:
        top_5_at_risk = at_risk_df.sort_values("PurchaseAmount", ascending=False).head(5)
        for _, r in top_5_at_risk.iterrows():
            top_at_risk_list.append(
                f"Customer ID {r['CustomerID']} ({r['label']}, {r['CategoryGroup']}, {r['RetailChannel']}): "
                f"${r['PurchaseAmount']:,.2f} (Satisfaction: {r['CustomerSatisfaction']}/5)"
            )
    top_at_risk_str = (
        "<br>• " + "<br>• ".join(top_at_risk_list) if top_at_risk_list else "None"
    )

    stable_df = df_filtered[df_filtered["label"] == "Stable"]
    stable_rev = stable_df["PurchaseAmount"].sum()
    stable_custs = stable_df["CustomerID"].nunique()
    stable_avg = stable_df["PurchaseAmount"].mean() if not stable_df.empty else 0.0

    # Channel satisfaction with and without Decline (Item 11b)
    online_df = df_filtered[df_filtered["RetailChannel"] == "Online"]
    online_all_sat = online_df["CustomerSatisfaction"].mean() if not online_df.empty else 0.0
    online_no_dec = online_df[online_df["label"] != "Decline"]
    online_no_dec_sat = online_no_dec["CustomerSatisfaction"].mean() if not online_no_dec.empty else 0.0

    store_df = df_filtered[df_filtered["RetailChannel"] == "Physical Store"]
    store_all_sat = store_df["CustomerSatisfaction"].mean() if not store_df.empty else 0.0
    store_no_dec = store_df[store_df["label"] != "Decline"]
    store_no_dec_sat = store_no_dec["CustomerSatisfaction"].mean() if not store_no_dec.empty else 0.0

    gap_all = abs(online_all_sat - store_all_sat)
    gap_no_dec = abs(online_no_dec_sat - store_no_dec_sat)
    if gap_all > 0 and gap_no_dec < gap_all * 0.5:
        gap_note = (
            f"*Note: The channel gap is {gap_all:.2f} points overall and {gap_no_dec:.2f} excluding Decline-labeled "
            f"transactions. It shrinks by more than half, so the gap largely reflects segment mix.*"
        )
    else:
        gap_note = (
            f"*Note: The channel gap is {gap_all:.2f} points overall and {gap_no_dec:.2f} excluding Decline-labeled "
            f"transactions. The gap may partly reflect segment mix, but it does not disappear when Decline is removed.*"
        )

    st.markdown("#### 🎯 Strategic Action Cards")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### 1. Invest in High-Momentum Segments")
            st.markdown(
                escape_dollars(
                    f"**Focus Area:** Category expansion in **{top_pg_cat}** for Promising & Growth tiers.<br>"
                    f"**Supporting Evidence:** In this filtered sample of {n_records} transactions, Promising and Growth "
                    f"transactions generated **${pg_rev:,.2f}** ({pg_rev / total_rev * 100:.1f}% of filtered revenue) across "
                    f"**{pg_custs}** unique customer IDs.<br>"
                    f"**Recommended Action:** Allocate VIP loyalty incentives, early-access marketing promos, and category bundling "
                    f"around {top_pg_cat} to accelerate customer migration into top spenders."
                ),
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("#### 2. Retain At-Risk Customers")
            st.markdown(
                escape_dollars(
                    f"**Focus Area:** Proactive intervention for transactions labeled Decline or with satisfaction ≤ 2.<br>"
                    f"**Supporting Evidence:** In this filtered sample, **{len(at_risk_df)}** transactions across **{at_risk_custs}** customer IDs are flagged At-Risk, "
                    f"accounting for **${at_risk_rev:,.2f}** ({at_risk_rev / total_rev * 100:.1f}% of filtered revenue).<br>"
                    f"**Top 5 At-Risk Transactions by Value:**{top_at_risk_str}<br><br>"
                    f"**Recommended Action:** Deploy customer-care intervention protocol (Suggested pilot, to be tested: "
                    f"dedicated customer service outreach and service recovery courtesy credit)."
                ),
                unsafe_allow_html=True,
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### 3. Re-engage Touchpoints with Satisfaction Gaps")
            st.markdown(
                escape_dollars(
                    f"**Focus Area:** Friction resolution across retail touchpoints.<br>"
                    f"**Supporting Evidence:** In this sample, Online satisfaction averages **{online_all_sat:.2f} / 5.0** overall "
                    f"(**{online_no_dec_sat:.2f} / 5.0** excluding Decline), compared to **{store_all_sat:.2f} / 5.0** overall "
                    f"(**{store_no_dec_sat:.2f} / 5.0** excluding Decline) in Physical Stores.<br>"
                    f"{gap_note}<br>"
                    f"**Recommended Action:** Conduct operational audit of checkout friction, in-store staff service, and "
                    f"online order fulfillment to address root causes of friction."
                ),
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("#### 4. Upsell & Stabilize Base Customers")
            st.markdown(
                escape_dollars(
                    f"**Focus Area:** Frequency acceleration for Stable customer segment.<br>"
                    f"**Supporting Evidence:** In this sample, Stable-labeled transactions total **${stable_rev:,.2f}** across "
                    f"**{stable_custs}** customer IDs, with an average order value of **${stable_avg:,.2f}**.<br>"
                    f"**Recommended Action:** Launch replenishment reminders and omnichannel crossover promotions "
                    f"(Suggested pilot, to be tested: e.g., 'Buy online, pick up in store' incentives and personalized reminders)."
                ),
                unsafe_allow_html=True,
            )

    st.caption("Segment labels apply to transactions. A customer ID can appear in more than one segment, so customer ID counts across cards do not add up to the total.")
    st.markdown("#### 🧭 Decision-Maker Priority Matrix")
    # Compute priority from simple rule (Item 11d)
    # High: Active risk/decline friction OR revenue >= $3,000
    # Medium: Revenue between $1,000 and $2,999
    # Low: Revenue < $1,000 or unclassified
    unlabeled_df = df_filtered[df_filtered["label"] == "Unlabeled"]
    unlabeled_rev = unlabeled_df["PurchaseAmount"].sum()

    priority_data = [
        {
            "Segment": "Decline & Low Satisfaction",
            "Recommended Action": "High-priority outreach & CS resolution protocol to prevent permanent customer attrition",
            "Supporting Metric": f"In this sample: {len(at_risk_df)} at-risk transactions ({at_risk_custs} customer IDs), ${at_risk_rev:,.2f} revenue exposed",
            "Priority": "High" if (at_risk_rev > 0 or at_risk_custs > 0) else "Low",
        },
        {
            "Segment": "Promising & Growth",
            "Recommended Action": f"Increase marketing spend on high-performing {top_pg_cat} & personalized upsell bundles",
            "Supporting Metric": f"In this sample: ${pg_rev:,.2f} revenue across {pg_custs} customer IDs",
            "Priority": "High" if pg_rev >= 3000 else ("Medium" if pg_rev >= 1000 else "Low"),
        },
        {
            "Segment": "Stable",
            "Recommended Action": "Drive omnichannel cross-shopping incentives to increase basket size and visit frequency",
            "Supporting Metric": f"In this sample: ${stable_rev:,.2f} total revenue, avg ticket ${stable_avg:,.2f}",
            "Priority": "High" if stable_rev >= 3000 else ("Medium" if stable_rev >= 1000 else "Low"),
        },
        {
            "Segment": "Unlabeled",
            "Recommended Action": "Enrich transactional and demographic tracking to ensure 100% behavioral classification",
            "Supporting Metric": f"In this sample: {len(unlabeled_df)} unlabeled transaction(s) (${unlabeled_rev:,.2f} revenue)",
            "Priority": "High" if unlabeled_rev >= 3000 else ("Medium" if unlabeled_rev >= 1000 else "Low"),
        },
    ]
    st.dataframe(pd.DataFrame(priority_data), use_container_width=True, hide_index=True)
    st.caption("Priority rule: High = Segments with active decline/risk friction or filtered revenue ≥ $3,000; Medium = Revenue between $1,000 and $2,999; Low = Revenue < $1,000.")


# -----------------------------------------------------------------------------
# 9. TAB 5: AI CHATBOT (Groq API)
# -----------------------------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-120b"


def compute_id_facts(d: pd.DataFrame):
    """Returns (distinct IDs, rows, IDs with >1 row, IDs with >1 distinct real label)."""
    n_ids = d["CustomerID"].nunique()
    n_rows = len(d)
    per_id = d.groupby("CustomerID").size()
    repeat_ids = int((per_id > 1).sum())
    labeled = d[d["label"] != "Unlabeled"]
    multi_label_ids = int((labeled.groupby("CustomerID")["label"].nunique() > 1).sum())
    return n_ids, n_rows, repeat_ids, multi_label_ids


def build_context(df_filtered: pd.DataFrame, df_all: pd.DataFrame = None) -> str:
    """
    Build a comprehensive, precomputed text summary of the CURRENTLY FILTERED dataset
    to inject into the Groq model context. Precomputes all shares, averages labeled
    'per transaction' and 'per customer', and rankings by revenue, ticket, and satisfaction.
    """
    total_rev = df_filtered["PurchaseAmount"].sum()
    unique_custs = df_filtered["CustomerID"].nunique()
    total_trans = len(df_filtered)
    avg_sat = df_filtered["CustomerSatisfaction"].mean() if not df_filtered.empty else 0.0

    overall_avg_per_trans = (total_rev / total_trans) if total_trans > 0 else 0.0
    overall_avg_per_cust = (total_rev / unique_custs) if unique_custs > 0 else 0.0

    # 1. Segment metrics & shares
    seg_records = []
    for label, g in df_filtered.groupby("label"):
        s_rev = g["PurchaseAmount"].sum()
        s_cnt = len(g)
        s_cust = g["CustomerID"].nunique()
        s_sat = g["CustomerSatisfaction"].mean()
        s_avg_trans = s_rev / s_cnt if s_cnt > 0 else 0.0
        s_avg_cust = s_rev / s_cust if s_cust > 0 else 0.0
        rev_share = (s_rev / total_rev * 100) if total_rev > 0 else 0.0
        trans_share = (s_cnt / total_trans * 100) if total_trans > 0 else 0.0
        cust_share = (s_cust / unique_custs * 100) if unique_custs > 0 else 0.0
        seg_records.append({
            "segment": label,
            "revenue": s_rev,
            "revenue_share_pct": rev_share,
            "transactions": s_cnt,
            "transaction_share_pct": trans_share,
            "unique_customer_ids": s_cust,
            "customer_id_share_pct": cust_share,
            "avg_spend_per_transaction": s_avg_trans,
            "avg_spend_per_customer_id": s_avg_cust,
            "avg_satisfaction_per_transaction": s_sat,
        })
    seg_df = pd.DataFrame(seg_records)

    # Precomputed Segment Rankings (excluding Unlabeled if labeled exist)
    seg_ranked = seg_df[seg_df["segment"] != "Unlabeled"] if (seg_df["segment"] != "Unlabeled").any() else seg_df
    rev_ranking = seg_ranked.sort_values("revenue", ascending=False).reset_index(drop=True)
    rev_rank_str = "\n".join([
        f"  - Rank {i+1}: {r['segment']} -> Revenue: ${r['revenue']:,.2f} ({r['revenue_share_pct']:.1f}% of total revenue, across {r['unique_customer_ids']} customer IDs, {r['transactions']} transactions)"
        for i, r in rev_ranking.iterrows()
    ])

    ticket_ranking = seg_ranked.sort_values("avg_spend_per_transaction", ascending=False).reset_index(drop=True)
    ticket_rank_str = "\n".join([
        f"  - Rank {i+1}: {r['segment']} -> Average Spend per Transaction: ${r['avg_spend_per_transaction']:.2f} (Total Revenue: ${r['revenue']:,.2f} / {r['transactions']} transactions)"
        for i, r in ticket_ranking.iterrows()
    ])

    cust_spend_ranking = seg_ranked.sort_values("avg_spend_per_customer_id", ascending=False).reset_index(drop=True)
    cust_spend_rank_str = "\n".join([
        f"  - Rank {i+1}: {r['segment']} -> Average Spend per Customer ID: ${r['avg_spend_per_customer_id']:.2f} (Total Revenue: ${r['revenue']:,.2f} / {r['unique_customer_ids']} customer IDs)"
        for i, r in cust_spend_ranking.iterrows()
    ])

    sat_ranking = seg_ranked.sort_values("avg_satisfaction_per_transaction", ascending=False).reset_index(drop=True)
    sat_rank_str = "\n".join([
        f"  - Rank {i+1}: {r['segment']} -> Average Customer Satisfaction: {r['avg_satisfaction_per_transaction']:.2f} / 5.0"
        for i, r in sat_ranking.iterrows()
    ])

    # 2. Region metrics
    reg_records = []
    for reg, g in df_filtered.groupby("CustomerRegion"):
        r_rev = g["PurchaseAmount"].sum()
        r_cnt = len(g)
        r_cust = g["CustomerID"].nunique()
        r_sat = g["CustomerSatisfaction"].mean()
        reg_records.append(
            f"  - Region {reg}: Revenue ${r_rev:,.2f} ({(r_rev/total_rev*100):.1f}%), {r_cnt} transactions ({(r_cnt/total_trans*100):.1f}%), {r_cust} customer IDs, avg spend per transaction ${r_rev/r_cnt:.2f}, avg satisfaction {r_sat:.2f}/5.0"
        )

    # 3. Category Group metrics
    cat_records = []
    for cat, g in df_filtered.groupby("CategoryGroup"):
        c_rev = g["PurchaseAmount"].sum()
        c_cnt = len(g)
        c_cust = g["CustomerID"].nunique()
        cat_records.append(
            f"  - {cat}: Revenue ${c_rev:,.2f} ({(c_rev/total_rev*100):.1f}%), {c_cnt} transactions, {c_cust} customer IDs, avg spend per transaction ${c_rev/c_cnt:.2f}"
        )

    # 4. Channel metrics
    chan_records = []
    for chan, g in df_filtered.groupby("RetailChannel"):
        ch_rev = g["PurchaseAmount"].sum()
        ch_cnt = len(g)
        ch_cust = g["CustomerID"].nunique()
        ch_sat = g["CustomerSatisfaction"].mean()
        chan_records.append(
            f"  - {chan}: Revenue ${ch_rev:,.2f} ({(ch_rev/total_rev*100):.1f}%), {ch_cnt} transactions ({(ch_cnt/total_trans*100):.1f}%), {ch_cust} customer IDs, avg spend per transaction ${ch_rev/ch_cnt:.2f}, avg spend per customer ID ${ch_rev/ch_cust:.2f}, avg satisfaction {ch_sat:.2f}/5.0"
        )

    # 5. Customer ID facts (labels apply to transactions, IDs repeat)
    base = df_all if df_all is not None else df_filtered
    a_ids, a_rows, a_repeat, a_multi = compute_id_facts(base)
    f_ids, f_rows, f_repeat, f_multi = compute_id_facts(df_filtered)

    # 6. At-Risk & Watchlist summary
    at_risk = df_filtered[df_filtered["AtRisk"]]
    at_risk_rev = at_risk["PurchaseAmount"].sum()
    at_risk_custs = at_risk["CustomerID"].nunique()

    context_str = f"""### DATA INTEGRITY NOTES (READ FIRST):
- Segment labels (Promising, Growth, Stable, Decline) apply to individual TRANSACTIONS, not to fixed customers.
- Full dataset: {a_ids} distinct CustomerIDs across {a_rows} transactions. {a_repeat} CustomerIDs appear in more than one transaction, and {a_multi} of them carry different segment labels on different transactions.
- Current filtered view: {f_ids} customer IDs across {f_rows} transactions ({f_repeat} repeat IDs, {f_multi} with more than one label).
- The project brief mentions about 99 customers, but the file contains only {a_ids} distinct CustomerIDs.
- Segments are NOT mutually exclusive groups of customers: customer ID counts per segment overlap and do not add up to the total.
- Always call counts of people 'customer IDs', never 'customers'.

### OVERALL FILTERED DATASET TOTALS:
- Total Revenue: ${total_rev:,.2f}
- Total Transactions: {total_trans}
- Total Unique Customer IDs: {unique_custs}
- Overall Average Spend per Transaction: ${overall_avg_per_trans:.2f}
- Overall Average Spend per Customer ID: ${overall_avg_per_cust:.2f}
- Overall Average Customer Satisfaction: {avg_sat:.2f} / 5.0
- Date Range Covered: {df_filtered['TransactionDate'].dt.date.min()} to {df_filtered['TransactionDate'].dt.date.max()} (approximately 5-week window)

### PRECOMPUTED SEGMENT RANKINGS (USE STRICTLY FOR COMPARISONS):
- Segments Ranked by Total Revenue:
{rev_rank_str}

- Segments Ranked by Average Spend per Transaction (Ticket Size):
{ticket_rank_str}

- Segments Ranked by Average Spend per Customer ID:
{cust_spend_rank_str}

- Segments Ranked by Average Customer Satisfaction (Highest to Lowest):
{sat_rank_str}

### DETAILED BREAKDOWN BY SEGMENT:
{seg_df.to_string(index=False)}

### BREAKDOWN BY GEOGRAPHIC REGION:
{chr(10).join(reg_records)}

### BREAKDOWN BY PRODUCT CATEGORY GROUP:
{chr(10).join(cat_records)}

### BREAKDOWN BY RETAIL CHANNEL:
{chr(10).join(chan_records)}

### RISK & DECLINE SUMMARY:
- At-Risk Transactions (Decline label OR Satisfaction <= 2): {len(at_risk)} transactions across {at_risk_custs} customer IDs, totaling ${at_risk_rev:,.2f} ({at_risk_rev/total_rev*100:.1f}% of filtered revenue).
"""
    return context_str


def render_tab_chatbot(df_filtered: pd.DataFrame, df_all: pd.DataFrame = None):
    """
    Tab 5: AI Customer Intelligence Chatbot powered by Groq API.
    Reads GROQ_API_KEY from st.secrets without hard-coding.
    """
    st.markdown("### 🤖 NovaRetail AI Customer Intelligence Advisor")
    st.markdown(
        f"<small style='color: #64748b;'>Ask strategic questions directly to the <code>{MODEL_NAME}</code> model, grounded strictly in your current filtered dataset.</small>",
        unsafe_allow_html=True,
    )

    # Retrieve API key strictly from st.secrets with safe fallback
    groq_api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        elif "groq_api_key" in st.secrets:
            groq_api_key = st.secrets["groq_api_key"]
    except Exception:
        groq_api_key = None

    if not groq_api_key:
        st.info(
            "ℹ️ **Groq API Key Required**: To chat with the AI Customer Intelligence Assistant, "
            "please configure `GROQ_API_KEY` in `.streamlit/secrets.toml` (locally) or in the Streamlit Community Cloud "
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

    # Precompute dataset context for current filter selection
    dataset_context = build_context(df_filtered, df_all)

    # Expander: "What the chatbot sees" (Item 7)
    with st.expander("🔍 What the chatbot sees (Grounding Context)"):
        st.text(dataset_context)

    # Toolbar: Suggested Questions & Clear Chat button
    st.markdown("##### 💡 Suggested Questions & Chat Controls")
    s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns([1, 1, 1, 1, 0.6])
    suggested_prompt = None

    with s_col1:
        if st.button("📈 Top Revenue Segment?", use_container_width=True):
            suggested_prompt = "Which segment generates the most revenue, and what are their key characteristics?"
    with s_col2:
        if st.button("⚠️ Signs of Decline?", use_container_width=True):
            suggested_prompt = "Where are the biggest signs of customer decline in this data, and what channels or regions are most exposed?"
    with s_col3:
        if st.button("🎯 Category to Invest in?", use_container_width=True):
            suggested_prompt = "Which product category group should we invest in for maximum customer growth?"
    with s_col4:
        if st.button("🏬 Online vs. Stores?", use_container_width=True):
            suggested_prompt = "How do online and physical store channels compare in terms of revenue and customer satisfaction?"
    with s_col5:
        if st.button("🗑️ Clear chat", use_container_width=True):
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
            st.rerun()

    # Chat history container (shown above input box)
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(escape_dollars(msg["content"]))

    # User input handling (via chat input at the bottom of the tab or suggested button)
    input_prompt = st.chat_input("Ask a question about the customer intelligence data...")
    active_prompt = suggested_prompt if suggested_prompt else input_prompt

    if active_prompt:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": active_prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(escape_dollars(active_prompt))

        # Generate response using Groq
        if not groq_api_key:
            reply = (
                "⚠️ **API Key Missing**: The Groq API key has not been configured in `st.secrets['GROQ_API_KEY']`. "
                "Please add it to your `.streamlit/secrets.toml` file to activate AI querying."
            )
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(escape_dollars(reply))
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        else:
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing customer intelligence data..."):
                        try:
                            from groq import Groq

                            client = Groq(api_key=groq_api_key)

                            system_instruction = (
                                "You are the senior Customer Intelligence AI Advisor for Sophia Martinez, "
                                "Director of Customer Intelligence at NovaRetail.\n"
                                "Answer questions based STRICTLY on the filtered dataset context provided below.\n\n"
                                "STRICT INSTRUCTIONS:\n"
                                "1. Use ONLY numbers, percentages, shares, and rankings explicitly given in the context. Never compute new ratios, shares, or percentages.\n"
                                "2. ALWAYS clearly state whether an average is 'per transaction' or 'per customer'—never mix, confuse, or interchange them.\n"
                                "3. For ALL comparisons (e.g., highest revenue, highest satisfaction, ticket size), you MUST strictly use the precomputed rankings provided in the context.\n"
                                "4. Always cite specific figures (revenue in $, transaction count n, unique customer count, satisfaction out of 5.0).\n"
                                "5. Mention small sample sizes when relevant (note that the total dataset has only 100 transactions spanning ~5 weeks from Jan 15 to Feb 20, 2023).\n"
                                "6. If the provided context does not have enough information to answer the question, explicitly state so rather than hallucinating or speculating.\n"
                                "7. Format answers cleanly with concise bullet points and bold key numbers.\n"
                                "8. Segment labels apply to TRANSACTIONS, not to fixed customers. A customer ID can carry different labels on different transactions, so never describe segments as groups of distinct people.\n"
                                "9. Refer to counts of people as 'customer IDs' (never 'customers'), and note that customer ID counts across segments overlap."
                            )

                            full_prompt = f"{dataset_context}\n\nUSER QUESTION: {active_prompt}"

                            completion = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": system_instruction},
                                    {"role": "user", "content": full_prompt},
                                ],
                                temperature=0.2,
                            )
                            reply = completion.choices[0].message.content
                            st.markdown(escape_dollars(reply))
                            st.session_state.chat_messages.append(
                                {"role": "assistant", "content": reply}
                            )

                        except Exception as e:
                            err_msg = (
                                f"⚠️ **Error connecting to Groq API**: {str(e)}\n\n"
                                "Please check your API key, network connection, or rate limits."
                            )
                            st.error(err_msg)
                            st.session_state.chat_messages.append(
                                {"role": "assistant", "content": err_msg}
                            )

    # Caption under the chat (Item 8)
    st.caption("Answers are generated by AI from a summary of the currently filtered data. They can contain errors, so verify against the charts.")


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

    # 4. Header & Subtitle (Item 17: concise non-redundant header count)
    st.title("🛍️ NovaRetail Customer Intelligence Dashboard")
    st.markdown(
        f"**Executive Intelligence Dashboard for Sophia Martinez, Director of Customer Intelligence**  \n"
        f"Showing {df_filtered['CustomerID'].nunique()} of {df['CustomerID'].nunique()} customer IDs | {len(df_filtered)} of {len(df)} transactions"
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
        render_tab_chatbot(df_filtered, df)

    # 8. Data Notes & Definitions Expander (Item 12)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 Data Notes, Definitions & Analytical Methodology"):
        st.markdown(
            """
            ### 1. Risk & Watchlist Definitions
            - **At-Risk (`AtRisk = True`)**: Transactions flagged when labeled **Decline** OR when reporting **CustomerSatisfaction ≤ 2**. These transactions show acute dissatisfaction or declining engagement.
            - **Watchlist (Early Warning)**: Transactions from engaged segments (**Promising, Growth, Stable**) with **CustomerSatisfaction ≤ 3** (neutral or low). Designed to detect friction before it shows up as a Decline label.

            ### 2. Satisfaction Tiers (`SatisfactionBand`)
            - **Low (1–2)**: Significant customer friction requiring immediate service recovery.
            - **Neutral (3)**: Indifferent customer experience at risk of downward drift.
            - **High (4–5)**: Satisfied and highly engaged customer base.

            ### 3. Product Category Group Mapping Table
            The 35 raw `ProductCategory` values are standardized into 9 analytical groups:
            - **Electronics**: Electronics, Gaming
            - **Clothing & Fashion**: Clothing, Fashion, Fashion & Apparel, Fashion Accessories, Children's Clothing, Sportswear
            - **Groceries**: Groceries, Grocery, Grocery Items, Food & Beverages
            - **Home & Furniture**: Home Appliances, Home & Garden, Home Decor, Home Improvement, Furniture, Furniture & Decor, Gardening Tools
            - **Sports & Outdoors**: Sporting Goods, Sports Equipment, Sports & Outdoors, Outdoor Equipment
            - **Health & Beauty**: Health & Wellness, Health Supplements, Health & Beauty, Beauty Products, Beauty & Personal Care, Cosmetics
            - **Books**: Books, Books & Magazines
            - **Toys**: Toys, Toys & Games
            - **Other**: Office Supplies, Automotive (and any unmapped categories)

            ### 4. Treatment of Missing Label
            - Exactly 1 record (row index 97, CustomerID 12509) had a missing segment label. It was imputed as **"Unlabeled"**.
            - It is **included** in all overall KPI totals, channel shares, and revenue metrics, but **excluded** from comparative segment charts (e.g. segment comparison bar, segment heatmap) to preserve analytical integrity.

            ### 5. Time Window & Small-Sample Warning
            - **Time Window**: Transactions span **January 15, 2023 to February 20, 2023** (~5 weeks). Trend analyses represent short-term operational cadence, not seasonal or annual cycles.
            - **Small-Sample Warning**: The dataset comprises 100 total transactions across 34 unique customer IDs (see section 6). Subsets (e.g., segment × category × region combinations) frequently have sample sizes of n < 5. All averages and recommendations should be treated as directional indicators rather than definitive statistical conclusions.
            """
        )
        n_ids, n_rows, n_repeat, n_multi = compute_id_facts(df)
        st.markdown(
            f"""
            ### 6. Customer IDs and Segments
            - The file contains **{n_ids} distinct CustomerIDs** across **{n_rows} transactions**. The project brief mentions about 99 customers, but the data does not contain that many distinct IDs.
            - **{n_repeat} CustomerIDs** appear in more than one transaction, and **{n_multi}** of them carry **different segment labels on different transactions**.
            - Therefore the segment label describes a **transaction**, not a fixed customer. A customer ID can appear in more than one segment, so customer ID counts per segment overlap and do not add up to the total.
            - For this reason the dashboard reports **customer IDs** and **labeled transactions** instead of "customers" and uses transactions as the unit of analysis.
            """
        )


if __name__ == "__main__":
    main()
