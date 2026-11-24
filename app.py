"""Used Car Price Predictor — Streamlit dashboard (Project 3).

Loads the random forest trained in notebooks/02_price_model.ipynb and lets
a user estimate a car's resale price from its details, plus explore the
cleaned dataset from notebooks/01_eda.ipynb.
"""

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NUMERIC_FEATURES = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
CATEGORICAL_FEATURES = ["Brand", "Fuel_Type", "Transmission", "Owner_Type", "Location"]

# Colors from the project's validated categorical palette (slot 1 = blue, slot 2 = orange).
# Picked per Streamlit theme so charts stay legible in both light and dark mode.
COLORS = {
    "light": {"series": "#2a78d6", "accent": "#eb6834", "text": "#0b0b0b"},
    "dark": {"series": "#3987e5", "accent": "#d95926", "text": "#f0efec"},
}
MUTED = "#898781"  # axis/gridlines — same value in both modes per the palette


def theme_colors():
    base = st.get_option("theme.base") or "light"
    return COLORS["dark" if base == "dark" else "light"]


def style_fig(fig, title):
    c = theme_colors()
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=c["text"],
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=MUTED, gridwidth=0.5, zeroline=False, linecolor=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=MUTED, gridwidth=0.5, zeroline=False, linecolor=MUTED)
    return fig


@st.cache_data
def load_data():
    return pd.read_csv("data/cars_clean.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/price_model.joblib")


st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Used Car Price Predictor")
st.caption(
    "Trained on 5,806 cleaned used-car listings from India (Cars24, via Kaggle). "
    "See the README for the full analysis behind this app."
)

df = load_data()
model = load_model()
c = theme_colors()

predict_tab, explore_tab = st.tabs(["Predict a price", "Explore the data"])

# ---------------------------------------------------------------------------
# Tab 1: Predict a price
# ---------------------------------------------------------------------------
with predict_tab:
    st.subheader("Enter the car's details")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", sorted(df["Brand"].unique()), index=sorted(df["Brand"].unique()).index("Maruti"))
        location = st.selectbox("Location", sorted(df["Location"].unique()))
        year = st.slider("Year", 1998, 2019, 2015)
    with col2:
        fuel_type = st.selectbox("Fuel type", sorted(df["Fuel_Type"].unique()))
        transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))
        owner_type = st.selectbox("Owner type", ["First", "Second", "Third", "Fourth & Above"])
    kilometers = st.number_input("Kilometers driven", min_value=0, max_value=800_000, value=50_000, step=1_000)

    # Mileage, engine, power and seats aren't asked directly (most people don't know
    # their car's exact specs). Instead we estimate them from similar cars in the
    # data: same brand and fuel type, or just same brand if that's too few cars.
    match = df[(df["Brand"] == brand) & (df["Fuel_Type"] == fuel_type)]
    match_label = f"{brand} {fuel_type}"
    if len(match) < 5:
        match = df[df["Brand"] == brand]
        match_label = brand
    if len(match) < 5:
        match = df
        match_label = "all cars"

    mileage = match["Mileage"].median()
    engine = match["Engine"].median()
    power = match["Power"].median()
    seats = match["Seats"].median()

    st.caption(
        f"Assuming typical specs for a **{match_label}** ({len(match)} similar cars): "
        f"{engine:.0f} CC engine · {power:.0f} bhp · {mileage:.1f} kmpl · {seats:.0f} seats."
    )

    if st.button("Predict price", type="primary"):
        input_row = pd.DataFrame([{
            "Year": year,
            "Kilometers_Driven": kilometers,
            "Mileage": mileage,
            "Engine": float(engine),
            "Power": power,
            "Seats": float(seats),
            "Brand": brand,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Owner_Type": owner_type,
            "Location": location,
        }])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        predicted_price = model.predict(input_row)[0]

        st.metric(
            "Estimated resale price",
            f"₹{predicted_price:.2f} lakhs",
            help=f"Approximately ₹{predicted_price * 100_000:,.0f}. "
                 "Typical model error on unseen cars is about 1.5 lakhs.",
        )

        same_brand = df[df["Brand"] == brand]["Price"]
        comparison = same_brand if len(same_brand) >= 20 else df["Price"]
        label = f"same-brand ({brand})" if len(same_brand) >= 20 else "all cars"

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=comparison, marker_color=c["series"], nbinsx=30, name="Price"))
        fig.add_vline(
            x=predicted_price, line_color=c["accent"], line_width=2, line_dash="dash",
            annotation_text="Your estimate", annotation_position="top",
        )
        fig.update_xaxes(title="Price (lakhs)")
        fig.update_yaxes(title="Number of cars")
        st.plotly_chart(style_fig(fig, f"Where this estimate falls among {label} cars"), width="stretch")

        st.caption(
            "This model is most accurate on cheaper, common cars. Rare, expensive "
            "cars (luxury brands, high power) have less training data, so estimates "
            "for them are less reliable — see the README's Project 2 limitations."
        )

# ---------------------------------------------------------------------------
# Tab 2: Explore the data
# ---------------------------------------------------------------------------
with explore_tab:
    st.subheader("Filter the dataset")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        year_range = st.slider("Year range", 1998, 2019, (1998, 2019))
    with fcol2:
        fuel_filter = st.multiselect("Fuel type", sorted(df["Fuel_Type"].unique()), default=sorted(df["Fuel_Type"].unique()))

    filtered = df[
        df["Year"].between(year_range[0], year_range[1])
        & df["Fuel_Type"].isin(fuel_filter)
    ]

    mcol1, mcol2 = st.columns(2)
    mcol1.metric("Cars matching filters", f"{len(filtered):,}")
    mcol2.metric("Median price", f"₹{filtered['Price'].median():.2f} lakhs" if len(filtered) else "—")

    if len(filtered) == 0:
        st.warning("No cars match these filters. Try widening the year range or fuel types.")
    else:
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=filtered["Price"], marker_color=c["series"], nbinsx=40))
        hist_fig.update_xaxes(title="Price (lakhs)")
        hist_fig.update_yaxes(title="Number of cars")
        st.plotly_chart(style_fig(hist_fig, "Distribution of car prices"), width="stretch")

        by_year = filtered.groupby("Year")["Price"].median().reset_index()
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(
            x=by_year["Year"], y=by_year["Price"], mode="lines+markers",
            line=dict(color=c["series"], width=2), marker=dict(size=8),
        ))
        line_fig.update_xaxes(title="Year")
        line_fig.update_yaxes(title="Median price (lakhs)")
        st.plotly_chart(style_fig(line_fig, "Median price by year"), width="stretch")

        by_brand = filtered.groupby("Brand")["Price"].median().sort_values(ascending=True)
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=by_brand.values, y=by_brand.index, orientation="h", marker_color=c["series"],
        ))
        bar_fig.update_xaxes(title="Median price (lakhs)")
        bar_fig.update_yaxes(title="")
        bar_fig.update_layout(height=max(400, len(by_brand) * 22))
        st.plotly_chart(style_fig(bar_fig, "Median price by brand"), width="stretch")

        if st.checkbox("Show underlying data"):
            st.dataframe(filtered, width="stretch")
