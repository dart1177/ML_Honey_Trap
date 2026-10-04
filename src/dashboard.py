"""
Streamlit dashboard for real-time honeypot monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Page config
st.set_page_config(layout="wide", page_title="ML_Honey_trap")

def load_data() -> pd.DataFrame:
    """
    Load processed session data.

    Returns:
        pd.DataFrame: Session data.
    """
    data_path = "data/processed/sessions.csv"
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            logging.error(f"Failed to read {data_path}: {e}")
            st.warning("Failed to load processed data. Showing sample data instead.")
    else:
        st.warning("Processed data not found. Showing sample data instead.")
        
    # Return sample data if not found or error
    return pd.DataFrame({
        "session_id": ["s1", "s2", "s3", "s4"],
        "src_ip": ["192.168.1.1", "10.0.0.5", "172.16.0.2", "192.168.1.1"],
        "start_time": ["2023-10-01 10:00:00", "2023-10-01 10:05:00", "2023-10-01 10:10:00", "2023-10-01 10:15:00"],
        "threat_level": ["Scanner", "Amateur", "Advanced", "Scanner"],
        "command_count": [2, 15, 8, 1],
        "iocs_extracted": ["none", "malware.sh", "10.0.0.1", "none"]
    })

def main() -> None:
    """Main dashboard application."""
    st.title("ML_Honey_trap Dashboard")
    
    # Sidebar
    st.sidebar.header("Controls")
    if st.sidebar.button("Refresh Data"):
        st.rerun()
        
    threat_filter = st.sidebar.multiselect(
        "Filter by Threat Level",
        ["Scanner", "Amateur", "Advanced"],
        default=["Scanner", "Amateur", "Advanced"]
    )
    
    df = load_data()
    
    # Convert timestamps if available
    if "start_time" in df.columns:
        try:
            df["start_time"] = pd.to_datetime(df["start_time"])
        except Exception:
            pass
            
    # Apply filters
    if "threat_level" in df.columns:
        filtered_df = df[df["threat_level"].isin(threat_filter)]
    else:
        filtered_df = df
        
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Live Feed", "Distribution", "Top IPs", "Timeline", "IOCs"])
    
    with tab1:
        st.subheader("Recent Sessions")
        st.dataframe(filtered_df.sort_values(by="start_time", ascending=False) if "start_time" in filtered_df.columns else filtered_df, use_container_width=True)
        
    with tab2:
        st.subheader("Threat Level Distribution")
        if "threat_level" in filtered_df.columns:
            counts = filtered_df["threat_level"].value_counts().reset_index()
            counts.columns = ["threat_level", "count"]
            fig = px.pie(counts, values="count", names="threat_level", hole=0.3, color="threat_level",
                         color_discrete_map={"Scanner": "green", "Amateur": "orange", "Advanced": "red"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No threat level data available.")
            
    with tab3:
        st.subheader("Most Active Source IPs")
        if "src_ip" in filtered_df.columns:
            ip_counts = filtered_df["src_ip"].value_counts().head(10).reset_index()
            ip_counts.columns = ["src_ip", "count"]
            fig2 = px.bar(ip_counts, x="src_ip", y="count", title="Top 10 IPs")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No source IP data available.")
            
    with tab4:
        st.subheader("Sessions Over Time")
        if "start_time" in filtered_df.columns:
            timeline_df = filtered_df.set_index("start_time").resample("1H").size().reset_index(name="count")
            fig3 = px.line(timeline_df, x="start_time", y="count", title="Session Frequency")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No timeline data available.")
            
    with tab5:
        st.subheader("Extracted Indicators of Compromise (IOCs)")
        if "iocs_extracted" in filtered_df.columns:
            iocs = filtered_df[filtered_df["iocs_extracted"] != "none"][["session_id", "src_ip", "iocs_extracted"]]
            st.dataframe(iocs, use_container_width=True)
        else:
            st.info("No IOC data available.")

if __name__ == "__main__":
    st.info("To run this dashboard, execute: `streamlit run src/dashboard.py`")
    main()
