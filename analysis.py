import json
from pathlib import Path
import statistics
from datetime import datetime, timedelta
import emoji
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from collections import Counter


st.set_page_config(page_title="Fishing Dashboard", layout="wide", initial_sidebar_state="expanded")

def safe_emojize(shortcode):
    result = emoji.emojize(f":{shortcode}:")
    return result if result != f":{shortcode}:" else None

def load_data(path: Path):
    if not path.exists():
        st.error(f"Data file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for user, info in data.items():
        for c in info.get("catches", []):
            rows.append({
                "user": user,
                "fish": c.get("fish", 0),
                "rarity": c.get("rarity", "unknown"),
                "time": pd.to_datetime(c.get("time")) if c.get("time") else pd.NaT,
            })
    if not rows:
        return pd.DataFrame(columns=["user", "fish", "rarity", "time"])
    df = pd.DataFrame(rows)
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.day_name()
    df["week"] = df["time"].dt.isocalendar().week
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    return df


def user_summary(df: pd.DataFrame, user: str):
    d = df[df["user"] == user]
    if d.empty:
        return {}
    total_fishes = int(d["fish"].sum())
    catches_made = len(d)
    biggest = int(d["fish"].max())
    smallest = int(d["fish"].min())
    avg = float(d["fish"].mean())
    median = float(d["fish"].median())
    std = float(d["fish"].std(ddof=0)) if catches_made > 1 else 0.0
    tier_counts = d["rarity"].value_counts().to_dict()
    tier_perc = {k: f"{v / catches_made * 100:.1f}%" for k, v in tier_counts.items()}
    per_day = d.groupby("date").agg({"fish": ["sum", "count"]})
    per_day.columns = ["fish_sum", "catches"]
    top_catches = d.sort_values("fish", ascending=False).head(10)
    
    # Additional stats
    active_days = d["date"].nunique()
    first_catch = d["time"].min()
    last_catch = d["time"].max()
    favorite_hour = d["hour"].mode()[0] if not d["hour"].empty else None
    best_day = d.groupby("date")["fish"].sum().idxmax()
    best_day_amount = int(d.groupby("date")["fish"].sum().max())
    
    # Streak calculation
    dates = sorted(d["date"].unique())
    current_streak = 0
    max_streak = 0
    if dates:
        streak = 1
        for i in range(1, len(dates)):
            if dates[i] == dates[i-1] + timedelta(days=1):
                streak += 1
            else:
                max_streak = max(max_streak, streak)
                streak = 1
        max_streak = max(max_streak, streak)
        
        # Current streak
        today = pd.Timestamp.now().date()
        if dates[-1] == today or dates[-1] == today - timedelta(days=1):
            streak = 1
            for i in range(len(dates)-2, -1, -1):
                if dates[i+1] == dates[i] + timedelta(days=1):
                    streak += 1
                else:
                    break
            if dates[-1] == today:
                current_streak = streak
            elif dates[-1] == today - timedelta(days=1):
                current_streak = streak
    
    return {
        "total_fishes": total_fishes,
        "catches_made": catches_made,
        "biggest": biggest,
        "smallest": smallest,
        "avg": avg,
        "median": median,
        "std": std,
        "tier_counts": tier_counts,
        "tier_perc": tier_perc,
        "per_day": per_day.reset_index(),
        "top_catches": top_catches,
        "active_days": active_days,
        "first_catch": first_catch,
        "last_catch": last_catch,
        "favorite_hour": favorite_hour,
        "best_day": best_day,
        "best_day_amount": best_day_amount,
        "max_streak": max_streak,
        "current_streak": current_streak,
    }


def overall_aggregation(df: pd.DataFrame):
    total_fishes = int(df["fish"].sum())
    total_catches = len(df)
    tier_totals = df.groupby("rarity")["fish"].sum().to_dict()
    tier_counts = df["rarity"].value_counts().to_dict()
    tier_perc_by_catches = {k: v / total_catches * 100 for k, v in tier_counts.items()}
    
    # tier stats: max/min single-catch value per tier
    tier_stats_df = df.groupby("rarity")["fish"].agg(["max", "min", "mean", "median"]).reset_index()
    tier_stats = {row["rarity"]: {
        "max": int(row["max"]), 
        "min": int(row["min"]),
        "mean": float(row["mean"]),
        "median": float(row["median"])
    } for _, row in tier_stats_df.iterrows()}
    
    # per-user tier sums
    per_user_tier = df.pivot_table(index="user", columns="rarity", values="fish", aggfunc="sum", fill_value=0)
    per_user_sum = df.groupby("user")["fish"].sum()
    per_user_counts = df.groupby("user")["fish"].count()
    per_user_max = df.groupby("user")["fish"].max()
    per_user_min = df.groupby("user")["fish"].min()
    tier_max_per_user = per_user_tier.max().to_dict() if not per_user_tier.empty else {}
    tier_min_per_user = per_user_tier.replace(0, np.nan).min().dropna().to_dict() if not per_user_tier.empty else {}
    
    # Time-based stats
    hourly_catches = df.groupby("hour").size()
    weekday_catches = df.groupby("weekday").size()
    monthly_catches = df.groupby("month").size()
    
    # Efficiency metrics
    efficiency = {}
    for user in df["user"].unique():
        user_data = df[df["user"] == user]
        efficiency[user] = {
            "avg_per_catch": float(user_data["fish"].mean()),
            "catches_per_day": len(user_data) / user_data["date"].nunique(),
            "consistency": float(user_data["fish"].std()) if len(user_data) > 1 else 0
        }
    
    return {
        "total_fishes": total_fishes,
        "total_catches": total_catches,
        "tier_totals": tier_totals,
        "tier_counts": tier_counts,
        "tier_perc_by_catches": tier_perc_by_catches,
        "tier_stats": tier_stats,
        "per_user_sum": per_user_sum,
        "per_user_counts": per_user_counts,
        "per_user_max": per_user_max,
        "per_user_min": per_user_min,
        "tier_max_per_user": tier_max_per_user,
        "tier_min_per_user": tier_min_per_user,
        "per_user_tier": per_user_tier,
        "hourly_catches": hourly_catches,
        "weekday_catches": weekday_catches,
        "monthly_catches": monthly_catches,
        "efficiency": efficiency,
    }


def style():
    css = """
    <style>
    @import url('');
    html, body, [data-testid="stAppViewContainer"] {
      font-family: 'Century Gothic', CenturyGothic, AppleGothic, sans-serif;
      background: linear-gradient(90deg,#0f172a 0%, #071033 50%, #00121a 100%);
      color: #e6eef8;
    }
    .stButton>button {background-color:#ff7b54;color:white;border-radius:8px}
    .card {background: rgba(255,255,255,0.04); padding: 16px; border-radius: 12px}
    .metric {background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px}
    h1 {color: #ffd166}
    h2 {color: #ffa7a7}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def main():
    base = Path(__file__).parent
    data_path = base / "fish_dictionary_1.json"
    df = load_data(data_path)
    style()

    st.title("🎣 Fishing Dashboard — Community Stats")
    st.markdown("A beautiful real-time dashboard summarizing catches.")

    if df is None or df.empty:
        st.info("No data to show. Make sure `fish_dictionary.json` is next to this script.")
        return

    agg = overall_aggregation(df)

    # load raw json for metadata (trash trophies etc.)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    user_list = sorted(df["user"].unique())
    user_sel = st.sidebar.selectbox("Select user (or All)", ["All"] + user_list)
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.sidebar.caption("Pick a start and end date (inclusive).")

    # Filter by date
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        df_view = df[(df["date"] >= start) & (df["date"] <= end)]
    else:
        df_view = df

    # Only show global overview when 'All' is selected
    if user_sel == "All":
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        total_fishes = int(df_view["fish"].sum())
        total_catches = len(df_view)
        unique_users = df_view["user"].nunique()
        avg_per_catch = total_fishes / total_catches if total_catches > 0 else 0
        
        col1.metric("🐟 Total Fishes", f"{total_fishes:,}")
        col2.metric("🎣 Total Catches", f"{total_catches:,}")
        col3.metric("👥 Active Users", f"{unique_users}")
        col4.metric("📊 Avg/Catch", f"{avg_per_catch:.1f}")

        # Two column layout for main content
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("🏆 Leaderboard — Top 10 by Total Fishes")
            top_users = agg["per_user_sum"].sort_values(ascending=False).head(10)
            lb = top_users.reset_index()
            lb.columns = ["User", "Total Fishes"]
            lb.index = range(1, len(lb) + 1)
            st.table(lb)

            # NEW: Activity Heatmap
            st.subheader("📅 Activity Heatmap")
            activity_pivot = df_view.pivot_table(
                index=df_view["time"].dt.day_name(),
                columns=df_view["time"].dt.hour,
                values="fish",
                aggfunc="count",
                fill_value=0
            )
            # Reorder weekdays
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            activity_pivot = activity_pivot.reindex([d for d in weekday_order if d in activity_pivot.index])
            
            fig_heatmap = px.imshow(
                activity_pivot,
                labels=dict(x="Hour of Day", y="Day of Week", color="Catches"),
                color_continuous_scale="Viridis",
                aspect="auto"
            )
            fig_heatmap.update_layout(height=300)
            st.plotly_chart(fig_heatmap, use_container_width=True)

        with right_col:
            st.subheader("🎯 Tier Breakdown")
            tier_counts = df_view["rarity"].value_counts()
            fig = px.pie(
                names=tier_counts.index, 
                values=tier_counts.values, 
                color=tier_counts.index,
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            # NEW: Best Fishing Hours
            st.subheader("⏰ Best Fishing Hours")
            hourly = df_view.groupby("hour")["fish"].sum().reset_index()
            fig_hour = px.bar(
                hourly, 
                x="hour", 
                y="fish",
                color="fish",
                color_continuous_scale="Turbo",
                labels={"hour": "Hour", "fish": "Total Fish"}
            )
            fig_hour.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig_hour, use_container_width=True)

        # NEW: Timeline Analysis
        st.subheader("📈 Timeline Analysis")
        timeline_tab1, timeline_tab2, timeline_tab3 = st.tabs(["Daily", "Weekly", "Monthly"])
        
        with timeline_tab1:
            daily_data = df_view.groupby("date").agg({
                "fish": "sum",
                "user": "nunique"
            }).reset_index()
            daily_data.columns = ["Date", "Total Fish", "Active Users"]
            
            fig_daily = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Fish Caught per Day", "Active Users per Day"),
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_data["Date"], y=daily_data["Total Fish"], 
                          mode='lines+markers', name='Fish', line=dict(color='#00b4d8')),
                row=1, col=1
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_data["Date"], y=daily_data["Active Users"], 
                          mode='lines+markers', name='Users', line=dict(color='#f77f00')),
                row=2, col=1
            )
            
            fig_daily.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with timeline_tab2:
            weekly_data = df_view.groupby([df_view["year"], df_view["week"]])["fish"].sum().reset_index()
            weekly_data["week_label"] = weekly_data["year"].astype(str) + "-W" + weekly_data["week"].astype(str).str.zfill(2)
            
            fig_weekly = px.bar(
                weekly_data, 
                x="week_label", 
                y="fish",
                labels={"week_label": "Week", "fish": "Total Fish"},
                color="fish",
                color_continuous_scale="Viridis"
            )
            fig_weekly.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        with timeline_tab3:
            monthly_data = df_view.groupby([df_view["year"], df_view["month"]])["fish"].sum().reset_index()
            monthly_data["month_label"] = pd.to_datetime(monthly_data[["year", "month"]].assign(day=1)).dt.strftime("%Y-%m")
            
            fig_monthly = px.area(
                monthly_data, 
                x="month_label", 
                y="fish",
                labels={"month_label": "Month", "fish": "Total Fish"},
                color_discrete_sequence=["#06ffa5"]
            )
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)

        # FIXED: Per-tier percent leaders
        st.markdown("### 🎖️ Per-Tier Dominance Leaders")
        st.markdown("*Shows who owns the biggest percentage of each tier's total fish*")
        
        tier_percent_leaders = {}
        for tier in df_view["rarity"].unique():
            tier_data = df_view[df_view["rarity"] == tier]
            tier_total = tier_data["fish"].sum()
            if tier_total <= 0:
                continue
            
            user_tier_sums = tier_data.groupby("user")["fish"].sum()
            if user_tier_sums.empty:
                continue
            
            leader = user_tier_sums.idxmax()
            leader_amount = int(user_tier_sums.max())
            pct = (leader_amount / tier_total) * 100
            
            tier_percent_leaders[tier] = {
                "user": leader,
                "pct": pct,
                "amount": leader_amount,
                "tier_total": int(tier_total)
            }
        
        if tier_percent_leaders:
            df_tpl = pd.DataFrame([
                {
                    "Tier": k,
                    "Leader": v["user"],
                    "Dominance": f"{v['pct']:.1f}%",
                    "Leader's Fish": v["amount"],
                    "Tier Total": v["tier_total"]
                } for k, v in tier_percent_leaders.items()
            ])
            df_tpl = df_tpl.sort_values("Dominance", ascending=False)
            st.table(df_tpl)

        # NEW: Interesting Statistics Section
        st.markdown("### 🌟 Interesting Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            # Lucky Strike Analysis
            st.markdown("#### 🍀 Lucky Strike Analysis")
            top_catches = df_view.nlargest(5, "fish")[["user", "fish", "rarity", "time"]]
            top_catches["time"] = top_catches["time"].dt.strftime("%Y-%m-%d %H:%M")
            top_catches.columns = ["User", "Fish", "Rarity", "Time"]
            st.table(top_catches)
            
            # Consistency Champions
            st.markdown("#### 📊 Most Consistent Fishers")
            consistency_data = []
            for user in df_view["user"].unique():
                user_data = df_view[df_view["user"] == user]
                if len(user_data) > 5:  # Only consider users with 5+ catches
                    std_dev = user_data["fish"].std()
                    avg = user_data["fish"].mean()
                    cv = (std_dev / avg * 100) if avg > 0 else 0  # Coefficient of variation
                    consistency_data.append({
                        "User": user,
                        "Avg Fish": f"{avg:.1f}",
                        "Consistency": f"{100 - cv:.1f}%",
                        "Catches": len(user_data)
                    })
            
            if consistency_data:
                consistency_df = pd.DataFrame(consistency_data).sort_values("Consistency", ascending=False).head(5)
                st.table(consistency_df)
        
        with insight_col2:
            # Time-based insights
            st.markdown("#### ⏱️ Timing Patterns")
            
            # Most productive hour
            best_hour = df_view.groupby("hour")["fish"].sum().idxmax()
            best_hour_total = df_view.groupby("hour")["fish"].sum().max()
            
            # Most productive weekday
            best_weekday = df_view.groupby("weekday")["fish"].sum().idxmax()
            best_weekday_total = df_view.groupby("weekday")["fish"].sum().max()
            
            # Busiest day ever
            busiest_day = df_view.groupby("date")["fish"].sum().idxmax()
            busiest_day_amount = df_view.groupby("date")["fish"].sum().max()
            
            st.metric("🕐 Golden Hour", f"{best_hour}:00 ({best_hour_total:,} fish)")
            st.metric("📅 Best Weekday", f"{best_weekday} ({best_weekday_total:,} fish)")
            st.metric("🏅 Record Day", f"{busiest_day} ({busiest_day_amount:,} fish)")
            
            # Rarity luck distribution
            st.markdown("#### 🎲 Rarity Distribution by User Count")
            rarity_by_users = df_view.groupby("rarity")["user"].nunique().reset_index()
            rarity_by_users.columns = ["Rarity", "Users Who Caught"]
            st.table(rarity_by_users)

        # NEW: Efficiency Rankings
        st.markdown("### ⚡ Efficiency Rankings")
        eff_col1, eff_col2, eff_col3 = st.columns(3)
        
        with eff_col1:
            st.markdown("#### Fish per Catch")
            eff_data = []
            for user in df_view["user"].unique():
                user_data = df_view[df_view["user"] == user]
                if len(user_data) >= 5:  # Minimum 5 catches
                    eff_data.append({
                        "User": user,
                        "Avg/Catch": user_data["fish"].mean(),
                        "Catches": len(user_data)
                    })
            if eff_data:
                eff_df = pd.DataFrame(eff_data).sort_values("Avg/Catch", ascending=False).head(5)
                eff_df["Avg/Catch"] = eff_df["Avg/Catch"].apply(lambda x: f"{x:.1f}")
                st.table(eff_df)
        
        with eff_col2:
            st.markdown("#### Fish per Active Day")
            daily_eff = []
            for user in df_view["user"].unique():
                user_data = df_view[df_view["user"] == user]
                active_days = user_data["date"].nunique()
                if active_days >= 3:  # Minimum 3 active days
                    total_fish = user_data["fish"].sum()
                    daily_eff.append({
                        "User": user,
                        "Fish/Day": total_fish / active_days,
                        "Days": active_days
                    })
            if daily_eff:
                daily_eff_df = pd.DataFrame(daily_eff).sort_values("Fish/Day", ascending=False).head(5)
                daily_eff_df["Fish/Day"] = daily_eff_df["Fish/Day"].apply(lambda x: f"{x:.1f}")
                st.table(daily_eff_df)
        
        with eff_col3:
            st.markdown("#### Catch Frequency")
            freq_data = []
            for user in df_view["user"].unique():
                user_data = df_view[df_view["user"] == user]
                active_days = user_data["date"].nunique()
                if active_days >= 3:
                    catches = len(user_data)
                    freq_data.append({
                        "User": user,
                        "Catches/Day": catches / active_days,
                        "Total": catches
                    })
            if freq_data:
                freq_df = pd.DataFrame(freq_data).sort_values("Catches/Day", ascending=False).head(5)
                freq_df["Catches/Day"] = freq_df["Catches/Day"].apply(lambda x: f"{x:.2f}")
                st.table(freq_df)

        # Trash trophies section
        st.markdown("### 🗑️ Trash Trophies — Community Collection")
        all_trash = Counter()
        owners = {}
        for u, info in raw_data.items():
            tc = info.get("trash_collected") or {}
            for item, cnt in tc.items():
                all_trash[item] += cnt
                owners.setdefault(item, set()).add(u)

        if all_trash:
            df_trash = pd.DataFrame([
                {
                    "Emoji": safe_emojize(k),
                    "Item": k,
                    "Total": v,
                    "Collectors": ", ".join(owners.get(k, [])),
                    "Top Collector": max(
                        [(u, raw_data[u].get("trash_collected", {}).get(k, 0)) for u in owners.get(k, [])],
                        key=lambda x: x[1]
                    )[0] if k in owners else ""
                } for k, v in all_trash.items()
            ])
            df_trash = df_trash.sort_values("Total", ascending=False)
            st.table(df_trash)
        else:
            st.write("No trash trophies found across users.")

    # User-specific view
    else:
        us = user_summary(df_view, user_sel)
        st.markdown(f"## 👤 User Profile: **{user_sel}**")
        
        # Main metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("🐟 Total", f"{us.get('total_fishes', 0):,}")
        col2.metric("🎣 Catches", f"{us.get('catches_made', 0)}")
        col3.metric("🏆 Best", f"{us.get('biggest', 0)}")
        col4.metric("📊 Avg", f"{us.get('avg', 0):.1f}")
        col5.metric("📅 Days", f"{us.get('active_days', 0)}")
        col6.metric("🔥 Streak", f"{us.get('current_streak', 0)}")
        
        # Additional user stats
        st.markdown("### 📈 Performance Statistics")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.metric("Median Catch", f"{us.get('median', 0):.1f}")
            st.metric("Standard Deviation", f"{us.get('std', 0):.1f}")
            st.metric("Smallest Catch", f"{us.get('smallest', 0)}")
        
        with stat_col2:
            first = us.get('first_catch')
            last = us.get('last_catch')
            if first:
                st.metric("First Catch", first.strftime("%Y-%m-%d"))
            if last:
                st.metric("Last Catch", last.strftime("%Y-%m-%d"))
            st.metric("Max Day Streak", f"{us.get('max_streak', 0)} days")
        
        with stat_col3:
            st.metric("Best Day", f"{us.get('best_day', 'N/A')}")
            st.metric("Best Day Amount", f"{us.get('best_day_amount', 0):,}")
            fav_hour = us.get('favorite_hour')
            if fav_hour is not None:
                st.metric("Favorite Hour", f"{fav_hour}:00")

        # Tier distribution
        st.markdown("### 🎯 Tier Distribution")
        tc = us.get("tier_counts", {})
        if tc:
            tier_col1, tier_col2 = st.columns([2, 1])
            with tier_col1:
                fig = px.bar(
                    x=list(tc.keys()), 
                    y=list(tc.values()), 
                    labels={"x": "Tier", "y": "Count"},
                    color=list(tc.keys()), 
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with tier_col2:
                tier_table = pd.DataFrame([
                    {"Tier": k, "Count": v, "Percentage": us.get("tier_perc", {}).get(k, "0%")}
                    for k, v in tc.items()
                ])
                st.table(tier_table)

        # Timeline
        st.markdown("### 📅 Fishing Timeline")
        per_day = us.get("per_day", pd.DataFrame())
        if not per_day.empty:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Fish per Day", "Catches per Day"),
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=per_day["date"], y=per_day["fish_sum"], 
                          mode='lines+markers', name='Fish', 
                          line=dict(color='#00b4d8', width=2),
                          marker=dict(size=8)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=per_day["date"], y=per_day["catches"], 
                       name='Catches', marker_color='#f77f00'),
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Top catches
        st.markdown("### 🏆 Top 10 Catches")
        tc = us.get("top_catches")
        if tc is not None and not tc.empty:
            tc_display = tc[["fish", "rarity", "time"]].copy()
            tc_display["time"] = tc_display["time"].dt.strftime("%Y-%m-%d %H:%M")
            tc_display.columns = ["Fish", "Rarity", "Time"]
            tc_display.index = range(1, len(tc_display) + 1)
            st.table(tc_display)

        # Hour distribution for user
        st.markdown("### ⏰ Fishing Hours Pattern")
        user_hourly = df_view[df_view["user"] == user_sel].groupby("hour")["fish"].agg(["sum", "count"]).reset_index()
        if not user_hourly.empty:
            fig_hours = go.Figure()
            fig_hours.add_trace(go.Bar(
                x=user_hourly["hour"],
                y=user_hourly["sum"],
                name="Total Fish",
                marker_color='lightblue',
                yaxis='y',
            ))
            fig_hours.add_trace(go.Scatter(
                x=user_hourly["hour"],
                y=user_hourly["count"],
                name="Catches",
                line=dict(color='red', width=2),
                yaxis='y2',
            ))
            fig_hours.update_layout(
                xaxis=dict(title="Hour of Day", dtick=1),
                yaxis=dict(title="Total Fish", side='left'),
                yaxis2=dict(title="Number of Catches", overlaying='y', side='right'),
                hovermode='x',
                height=350
            )
            st.plotly_chart(fig_hours, use_container_width=True)

        # Weekday pattern for user
        st.markdown("### 📅 Weekday Performance")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        user_weekday = df_view[df_view["user"] == user_sel].groupby("weekday")["fish"].agg(["sum", "count", "mean"]).reset_index()
        user_weekday["weekday"] = pd.Categorical(user_weekday["weekday"], categories=weekday_order, ordered=True)
        user_weekday = user_weekday.sort_values("weekday")
        
        if not user_weekday.empty:
            fig_weekday = px.bar(
                user_weekday,
                x="weekday",
                y="sum",
                color="mean",
                color_continuous_scale="Viridis",
                labels={"weekday": "Day", "sum": "Total Fish", "mean": "Avg per Catch"},
                text="count"
            )
            fig_weekday.update_traces(texttemplate='%{text} catches', textposition='outside')
            fig_weekday.update_layout(height=350)
            st.plotly_chart(fig_weekday, use_container_width=True)

        # User's trash trophies
        st.markdown("### 🗑️ Trash Collection")
        user_trash = raw_data.get(user_sel, {}).get("trash_collected") or {}
        if user_trash:
            df_ut = pd.DataFrame(list(user_trash.items()), columns=["Item", "Count"]).sort_values("Count", ascending=False)
            df_ut["Emoji"] = df_ut["Item"].map(lambda k: safe_emojize(k))
            df_ut = df_ut[["Emoji", "Item", "Count"]]
            df_ut.index = range(1, len(df_ut) + 1)
            st.table(df_ut)
        else:
            st.info("No trash trophies collected yet. Keep fishing!")

        # Comparison with community
        st.markdown("### 📊 Community Comparison")
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            # User ranking
            all_users_sum = agg["per_user_sum"].sort_values(ascending=False)
            user_rank = list(all_users_sum.index).index(user_sel) + 1 if user_sel in all_users_sum.index else None
            
            if user_rank:
                st.metric("Overall Rank", f"#{user_rank} of {len(all_users_sum)}")
                
                # Percentile
                percentile = ((len(all_users_sum) - user_rank + 1) / len(all_users_sum)) * 100
                st.metric("Percentile", f"Top {100-percentile:.1f}%")
            
            # Average comparison
            community_avg = df_view["fish"].mean()
            user_avg = us.get("avg", 0)
            diff_pct = ((user_avg - community_avg) / community_avg * 100) if community_avg > 0 else 0
            
            st.metric("Your Avg vs Community", f"{user_avg:.1f} vs {community_avg:.1f}",
                     delta=f"{diff_pct:+.1f}%")
        
        with comp_col2:
            # Tier specialization
            st.markdown("#### Tier Specialization")
            
            # Extract tier counts from the DataFrame properly
            if isinstance(tc, pd.DataFrame) and not tc.empty:
                # tc appears to be the user's fishing data, so let's extract tier counts
                tier_counts = tc["rarity"].value_counts().to_dict()
            elif isinstance(tc, dict):
                tier_counts = tc
            else:
                tier_counts = {}
            
            if tier_counts:
                user_tier_focus = max(tier_counts.items(), key=lambda x: x[1])
                st.metric("Most Caught Tier", f"{user_tier_focus[0]} ({user_tier_focus[1]} catches)")
                
                # Check if user is leader in any tier
                tier_leaderships = []
                for tier in tier_counts.keys():
                    tier_data = df_view[df_view["rarity"] == tier]
                    if not tier_data.empty:
                        tier_by_user = tier_data.groupby("user")["fish"].sum()
                        if user_sel in tier_by_user.index and tier_by_user[user_sel] == tier_by_user.max():
                            tier_leaderships.append(tier)
                
                if tier_leaderships:
                    st.success(f"🏆 Tier Leader in: {', '.join(tier_leaderships)}")
            else:
                st.info("No tier data available for this user.")

    # Footer
    st.markdown("---")
    st.caption("Dashboard generated from `fish_dictionary.json`. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    
main()